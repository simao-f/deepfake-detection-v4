import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import time
import copy
import logging
import random
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import default_collate

# Modern Mixed Precision (Fixes the deprecation warnings you saw earlier)
from torch.cuda.amp import autocast, GradScaler 

# --- Reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- Logging Setup ---
def setup_logging(model_name="vit"):
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2] # Adjusts to ../../
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

# --- Robust Data Loading ---
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index): # type: ignore
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (UnidentifiedImageError, OSError) as e:
            logging.warning(f"Skipping corrupted image {path}: {e}")
            return None # Return None instead of a black image
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

def safe_collate(batch):
    # Filter out None values from the batch
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]) 
    return default_collate(batch)

# --- Training Engine ---
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25, patience=5):
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Early Stopping Tracking
    epochs_no_improve = 0
    
    # Initialize Scaler for L4 GPU Acceleration
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

            # TQDM Progress Bar
            pbar = tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch + 1}", leave=False)

            for batch in pbar:
                if not batch or len(batch) == 0:
                    continue
                    
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # MIXED PRECISION CONTEXT (Faster!)
                with torch.set_grad_enabled(phase == 'Train'):
                    with autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'Train':
                        # Scale loss for stability in half-precision
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep Copy Logic & Early Stopping
            if phase == 'Validation':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                # Step the scheduler
                scheduler.step(epoch_loss)
                val_acc_history.append(epoch_acc)

        # Early Stopping Check
        if epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    time_elapsed = time.time() - since
    logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def main():
    set_seed(42)
    setup_logging("vit")
    
    # Path Setup using Pathlib (More robust)
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parents[1] / "data" / "Dataset"
    
    logging.info(f"Looking for data in: {data_dir}")

    # Data Transforms
    # ViT needs 224x224. 
    # Added RandomResizedCrop for better generalization.
    data_transforms = {
        'Train': transforms.Compose([
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

    # Use SafeImageFolder to skip bad files
    image_datasets = {x: SafeImageFolder(str(data_dir / x), data_transforms[x])
                      for x in ['Train', 'Validation']}
    
    # Determine Batch Size
    # ViT is heavy. 128 might be too big for 1 GPU, but fine for 4 GPUs (32 per GPU).
    base_batch_size = 32
    gpu_count = torch.cuda.device_count()
    batch_size = base_batch_size * (gpu_count if gpu_count > 0 else 1)
    
    logging.info(f"Using Batch Size: {batch_size} (distributed over {gpu_count} GPUs)")

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                  batch_size=batch_size,
                                                  shuffle=True, 
                                                  num_workers=8, # Increased workers for faster feeding
                                                  pin_memory=True, # Faster CPU->GPU transfer
                                                  collate_fn=safe_collate)
                  for x in ['Train', 'Validation']}
    
    class_names = image_datasets['Train'].classes
    logging.info(f"Classes: {class_names}")

    # Initialize ViT
    logging.info("Initializing Vision Transformer (ViT-B/16)...")
    model_ft = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    # Modify Head for Binary Classification
    # ViT structure: heads -> head (Linear)
    # Explicitly check type to satisfy linter
    head_layer = model_ft.heads.head
    if isinstance(head_layer, nn.Linear):
        in_features = head_layer.in_features
    else:
        raise ValueError("Expected model_ft.heads.head to be nn.Linear")
        
    model_ft.heads.head = nn.Linear(in_features, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        logging.info(f"Wrapping model in DataParallel for {torch.cuda.device_count()} GPUs.")
        model_ft = nn.DataParallel(model_ft)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Use AdamW instead of SGD (Standard for Transformers)
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=2)

    model_ft, hist = train_model(
        model_ft, dataloaders, criterion, optimizer_ft, scheduler, device, 
        num_epochs=15, patience=4
    )

    # Clean Save
    save_path = current_dir / 'vit_deepfake_detection.pth'
    
    # Unwrap DataParallel before saving
    model_state = model_ft.module.state_dict() if isinstance(model_ft, nn.DataParallel) else model_ft.state_dict()
    
    torch.save(model_state, save_path)
    logging.info(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()