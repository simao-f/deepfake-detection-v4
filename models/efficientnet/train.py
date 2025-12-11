import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
from PIL import UnidentifiedImageError, Image
import numpy as np
import logging
from datetime import datetime

# --- Logging Setup ---
def setup_logging(rank, model_name="efficientnet"):
    if rank != 0:
        return None
        
    # Get project root (assuming script is in models/efficientnet/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_train_{timestamp}.log")
    
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

# --- DDP Setup ---
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        return local_rank, rank, world_size
    else:
        # Fallback for single GPU debugging
        print("DDP environment variables not found. Running in single-process mode.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# --- Custom Dataset with Albumentations Support (Optional but recommended for complex augs) ---
# For simplicity and speed, we will stick to torchvision transforms as requested, 
# but implement the specific augmentations: JPEG compression simulation, Noise.
# Note: Torchvision doesn't have direct "JPEG Compression" or "Gaussian Noise" transforms easily accessible 
# without custom lambdas or external libraries like Albumentations. 
# We will implement simple custom transforms for these.

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index): # type: ignore
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (UnidentifiedImageError, OSError):
            print(f"Warning: Skipping corrupted image {path}")
            return None
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)

def get_model(device, local_rank=None):
    """
    Loads a pre-trained EfficientNet-B0 model and modifies the classifier 
    for binary classification (Real vs Fake).
    """
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    
    # Modify the classifier head for Binary Classification (1 output)
    # Ensure in_features is an int
    # Access the Linear layer specifically to satisfy type checker
    classifier_layer = model.classifier[1]
    if isinstance(classifier_layer, nn.Linear):
        in_features = classifier_layer.in_features
    else:
        # Fallback or error if structure is unexpected
        raise ValueError("Expected model.classifier[1] to be nn.Linear")
        
    model.classifier[1] = nn.Linear(in_features, 1)
    
    model = model.to(device)
    
    if dist.is_initialized():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])
        
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, rank):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Only show progress bar on rank 0
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", disable=(rank != 0))
    
    for batch in pbar:
        if batch is None:
            continue
        inputs, labels = batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad()

        # Mixed Precision Training
        with autocast(dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistics
        preds = torch.sigmoid(outputs) > 0.5
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        if rank == 0:
            pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / total_samples
    if isinstance(running_corrects, torch.Tensor):
        epoch_acc = running_corrects.float() / total_samples
    else:
        epoch_acc = float(running_corrects) / total_samples
    
    # Sync metrics across processes for logging (optional, but good for accuracy)
    if dist.is_initialized():
        metrics = torch.tensor([epoch_loss, epoch_acc], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= dist.get_world_size()
        epoch_loss, epoch_acc = metrics[0].item(), metrics[1].item()

    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, epoch, rank):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs) > 0.5

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    if isinstance(running_corrects, torch.Tensor):
        epoch_acc = running_corrects.float() / total_samples
    else:
        epoch_acc = float(running_corrects) / total_samples
    
    if dist.is_initialized():
        metrics = torch.tensor([epoch_loss, epoch_acc], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= dist.get_world_size()
        epoch_loss, epoch_acc = metrics[0].item(), metrics[1].item()
        
    return epoch_loss, epoch_acc

def main():
    # 1. DDP Setup
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    
    if rank == 0:
        setup_logging(rank, "efficientnet_b0")
        logging.info(f"Starting training on {world_size} GPUs.")
        logging.info(f"Device: {device}")

    # 2. Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(script_dir, "../../data/Dataset")
    BATCH_SIZE = 64 # Per GPU batch size
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    
    # 3. Data Augmentation
    # Deepfake specific: Noise, Compression (simulated via lower quality resize or custom)
    # We use standard torchvision here + Gaussian Noise
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.02), # Add slight noise
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Datasets & Dataloaders
    train_dataset = SafeImageFolder(os.path.join(DATA_DIR, 'Train'), train_transforms)
    val_dataset = SafeImageFolder(os.path.join(DATA_DIR, 'Validation'), val_transforms)
    
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=safe_collate
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=safe_collate
    )

    # 5. Model, Loss, Optimizer, Scaler
    model = get_model(device, local_rank)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler() # For AMP

    # 6. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        if dist.is_initialized() and train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, rank)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, rank)
        
        if rank == 0:
            logging.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            # Save Best Model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(script_dir, "efficientnet_b0_deepfake_best.pth")
                # Save underlying model (unwrap DDP)
                model_to_save = model.module if hasattr(model, 'module') else model
                if isinstance(model_to_save, (nn.Module, DDP)):
                    torch.save(model_to_save.state_dict(), save_path)
                    logging.info(f"Saved best model to {save_path}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
