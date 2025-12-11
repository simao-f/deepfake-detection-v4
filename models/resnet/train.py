import os
import time
import copy
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import UnidentifiedImageError
from torch.cuda.amp import GradScaler, autocast # Key for speed on L4 GPUs
from torch.utils.data import default_collate
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet50_Weights

# --- Configuration & Reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True # Uncomment for strict reproducibility (slower)
    # torch.backends.cudnn.benchmark = False

def setup_logging(model_name="resnet50"):
    # Use Pathlib for robust paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2] # Go up two levels
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{model_name}_train_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Saving logs to {log_file}")
    return log_file

# --- Data Handling ---
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index): # type: ignore
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (UnidentifiedImageError, OSError) as e:
            logging.warning(f"Skipping corrupted image {path}: {e}")
            return None
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]) # Return empty tensor if whole batch is bad
    return default_collate(batch)

# --- Model Setup ---
def get_model(device):
    logging.info("Initializing ResNet-50...")
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    
    # Unfreeze all layers (Fine-tuning)
    # If dataset is small, uncomment loop below to freeze backbone
    # for param in model.parameters():
    #     param.requires_grad = False
    
    in_features = model.fc.in_features
    
    # Improved Head: Batch Norm helps training stability
    model.fc = nn.Sequential( # type: ignore
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 1)
    )
    
    model = model.to(device)
    return model

# --- Training Loop ---
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=10, patience=3):
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Early Stopping variables
    epochs_no_improve = 0

    # Initialize Scaler for Mixed Precision
    scaler = GradScaler()

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch + 1}/{num_epochs}')
        logging.info('-' * 10)

        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # Create progress bar
            pbar = tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch + 1}", leave=False)
            
            for batch in pbar:
                if not batch or len(batch) == 0:
                    continue
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()

                # MIXED PRECISION CONTEXT
                with torch.set_grad_enabled(phase == 'Train'):
                    # Autocast runs the forward pass in float16 where possible
                    with autocast():
                        outputs = model(inputs)
                        preds = torch.sigmoid(outputs) > 0.5
                        loss = criterion(outputs, labels)

                    if phase == 'Train':
                        # Scale loss and backprop
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                
                # Update progress bar description
                pbar.set_postfix({'loss': loss.item()})

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best one
            if phase == 'Validation':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0 # Reset counter
                else:
                    epochs_no_improve += 1
                
                # Step the scheduler based on validation metrics
                scheduler.step(epoch_loss)
                val_acc_history.append(epoch_acc)

        # Early Stopping Check
        if epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered! No improvement for {patience} epochs.")
            break

    time_elapsed = time.time() - since
    logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def main():
    set_seed(42)
    setup_logging("resnet50")
    
    # Path configuration
    script_dir = Path(__file__).resolve().parent
    # Assuming standard structure: project/data/Dataset
    DATA_DIR = script_dir.parents[1] / "data" / "Dataset"
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20 # Increased because we have early stopping now
    LEARNING_RATE = 1e-4 # Slightly lower starting LR is better for fine-tuning
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Apple Silicon).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        logging.info(f"Using CUDA with {gpu_count} GPUs.")
        if gpu_count > 1:
            BATCH_SIZE *= gpu_count
    else:
        device = torch.device("cpu")
        logging.info("Using CPU.")

    # Data Transforms
    data_transforms = {
        'Train': transforms.Compose([
            transforms.Resize(256),
            # RandomResizedCrop is better for generalization than CenterCrop
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    logging.info(f"Loading datasets from {DATA_DIR}...")
    
    # Ensure directories exist
    if not (DATA_DIR / "Train").exists():
        logging.error(f"Train directory not found at {DATA_DIR / 'Train'}")
        return

    image_datasets = {x: SafeImageFolder(str(DATA_DIR / x), data_transforms[x])
                      for x in ['Train', 'Validation']}
    
    # Use os.cpu_count() for num_workers
    num_workers = min(8, os.cpu_count() or 4) 
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True, 
                                                  num_workers=num_workers,
                                                  pin_memory=True, # Faster data transfer to GPU
                                                  collate_fn=safe_collate)
                  for x in ['Train', 'Validation']}
    
    class_names = image_datasets['Train'].classes
    logging.info(f"Classes: {class_names}")

    model = get_model(device)
    
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: Reduce LR if validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    model, hist = train_model(
        model, dataloaders, criterion, optimizer, scheduler, device, 
        num_epochs=NUM_EPOCHS, patience=5
    )

    save_path = script_dir / "resnet50_deepfake.pth"
    
    # Clean save for DataParallel
    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    torch.save({
        'model_state_dict': model_state_dict,
        'model_type': 'resnet',
        'class_names': class_names,
        'val_acc_history': hist
    }, save_path)
    logging.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()