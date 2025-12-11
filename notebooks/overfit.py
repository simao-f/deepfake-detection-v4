import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# --- Configuration ---
BATCH_SIZE = 64
NUM_WORKERS = 4
IMG_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["Fake", "Real"]

# Project Root (assuming script is in notebooks/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
MODEL_PATHS = {
    "EfficientNet": PROJECT_ROOT / "models/efficientnet/efficientnet_b0_deepfake_best.pth",
    "ResNet": PROJECT_ROOT / "models/resnet/resnet50_deepfake.pth",
    "ViT": PROJECT_ROOT / "models/vision_transformer/vit_deepfake_detection.pth"
}

DATA_DIRS = {
    "Train": PROJECT_ROOT / "data/Dataset/Train",
    "Validation": PROJECT_ROOT / "data/Dataset/Validation",
    "Test": PROJECT_ROOT / "data/Dataset/Test"
}

OUTPUT_DIR = PROJECT_ROOT / "logs/overfit_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "analysis.log")
    ]
)
logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Returns the appropriate torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_model(architecture: str, num_classes: int = 2) -> nn.Module:
    """
    Constructs the model architecture.
    """
    architecture = architecture.lower()
    
    if "efficientnet" in architecture:
        model = models.efficientnet_b0(weights=None)
        # EfficientNet classifier head is model.classifier[1]
        classifier = model.classifier[1]
        if isinstance(classifier, nn.Linear):
            in_features = classifier.in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            
    elif "resnet" in architecture:
        model = models.resnet50(weights=None)
        # ResNet classifier head is model.fc
        # Cast to Linear to satisfy type checker or ignore
        if isinstance(model.fc, nn.Linear):
            in_features = model.fc.in_features
            # Assign Sequential to fc (valid in PyTorch, but type checker might complain)
            model.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, num_classes)
            ) # type: ignore
        
    elif "vit" in architecture:
        model = models.vit_b_16(weights=None)
        # ViT classifier head is model.heads.head
        head = model.heads.head
        if isinstance(head, nn.Linear):
            in_features = head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
            
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
        
    return model

def load_model(name: str, path: Path, device: torch.device) -> Optional[nn.Module]:
    """
    Loads a model from a .pth file, handling state_dict keys and architecture.
    """
    logger.info(f"Loading {name} from {path}...")
    if not path.exists():
        logger.error(f"Model file not found: {path}")
        return None

    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as e:
        logger.error(f"Failed to load checkpoint for {name}: {e}")
        return None

    # Extract state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Clean state dict (remove 'module.' prefix from DDP)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Determine num_classes from the weights of the last layer
    num_classes = 1 # Default to binary (1 output neuron)
    
    # Heuristic to find the last weight layer
    last_layer_keys = [k for k in clean_state_dict.keys() if "weight" in k]
    classifier_keys = [k for k in last_layer_keys if "classifier" in k or "fc" in k or "head" in k]
    
    if classifier_keys:
        last_key = classifier_keys[-1] # Take the last one found
        output_dim = clean_state_dict[last_key].shape[0]
        # If output_dim is 1, it's binary. If 2, it's also binary but categorical.
        num_classes = output_dim
    
    logger.info(f"Detected {num_classes} output neurons for {name}.")

    try:
        model = build_model(name, num_classes)
        # Try strict loading first
        model.load_state_dict(clean_state_dict)
    except RuntimeError as e:
        logger.warning(f"Strict load failed for {name}: {e}. Retrying with strict=False.")
        try:
            model = build_model(name, num_classes)
            model.load_state_dict(clean_state_dict, strict=False)
        except Exception as e2:
            logger.error(f"Failed to load model {name} even with strict=False: {e2}")
            return None
        
    model.to(device)
    model.eval()
    return model

def evaluate(model: nn.Module, data_dir: Path, device: torch.device) -> Tuple[Optional[List], Optional[List]]:
    """
    Runs inference on a dataset and returns true labels and predictions.
    """
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return None, None

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    
    try:
        dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    except Exception as e:
        logger.error(f"Failed to create ImageFolder for {data_dir}: {e}")
        return None, None

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    y_true = []
    y_pred = []
    
    # Use AMP for faster inference on CUDA
    use_amp = device.type == "cuda"
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Eval {data_dir.name}", leave=False):
            inputs = inputs.to(device)
            
            if use_amp:
                with torch.amp.autocast('cuda'): # type: ignore
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            # Handle Binary (1 output) vs Categorical (2 outputs)
            if outputs.shape[1] == 1:
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
            else:
                _, preds = torch.max(outputs, 1)
                
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            
    return y_true, y_pred

def main():
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    
    results = []
    test_class_metrics = []
    full_logs = []
    
    for model_name, model_path in MODEL_PATHS.items():
        model = load_model(model_name, model_path, device)
        if model is None:
            continue
            
        # Explicitly type the dictionary to allow both str and float
        model_results: Dict[str, Union[str, float]] = {"Model": model_name}
        
        for split_name, split_path in DATA_DIRS.items():
            logger.info(f"Evaluating {model_name} on {split_name}...")
            y_true, y_pred = evaluate(model, split_path, device)
            
            if y_true is None or y_pred is None:
                model_results[f"{split_name} Acc"] = np.nan
                continue
                
            acc = accuracy_score(y_true, y_pred)
            model_results[f"{split_name} Acc"] = acc
            
            # Detailed Report
            # Cast the output of classification_report to the expected dictionary structure
            report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
            report_dict = cast(Dict[str, Dict[str, float]], report_dict)
            
            report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
            
            full_logs.append(f"=== {model_name} - {split_name} Set ===\n{report_str}\n")
            
            # For Test Set: Extract Class-wise metrics
            if split_name == "Test":
                test_class_metrics.append({
                    "Model": model_name,
                    "Fake Precision": report_dict["Fake"]["precision"],
                    "Fake Recall": report_dict["Fake"]["recall"],
                    "Real Precision": report_dict["Real"]["precision"],
                    "Real Recall": report_dict["Real"]["recall"]
                })
        
        results.append(model_results)

    # --- Reporting ---
    
    if not results:
        logger.error("No results generated. Check model paths and data directories.")
        return

    # 1. Master Table
    df_master = pd.DataFrame(results)
    # Reorder columns if they exist
    cols = ["Model", "Train Acc", "Validation Acc", "Test Acc"]
    existing_cols = [c for c in cols if c in df_master.columns]
    df_master = df_master[existing_cols]
    
    print("\n" + "="*60)
    print("  MASTER OVERFITTING ANALYSIS TABLE")
    print("="*60)
    print(df_master.to_markdown(index=False, floatfmt=".2%"))
    
    # 2. Test Set Breakdown
    if test_class_metrics:
        df_test = pd.DataFrame(test_class_metrics)
        print("\n" + "="*60)
        print("  TEST SET BIAS ANALYSIS (Real vs Fake)")
        print("="*60)
        print(df_test.to_markdown(index=False, floatfmt=".2%"))
        
    # 3. Save Files
    df_master.to_csv(OUTPUT_DIR / "model_comparison_summary.csv", index=False)
    with open(OUTPUT_DIR / "full_evaluation_logs.txt", "w") as f:
        f.writelines(full_logs)
        
    logger.info(f"\nReports saved to {OUTPUT_DIR}")
    
    # 4. Overfitting Analysis
    print("\n" + "="*60)
    print("  OVERFITTING DIAGNOSIS")
    print("="*60)
    
    for _, row in df_master.iterrows():
        train_acc = row.get("Train Acc", 0)
        test_acc = row.get("Test Acc", 0)
        
        if pd.notna(train_acc) and pd.notna(test_acc):
            diff = train_acc - test_acc
            if diff > 0.05:
                print(f"⚠️  WARNING: {row['Model']} is OVERFITTING.")
                print(f"   Train: {train_acc:.2%} | Test: {test_acc:.2%} | Gap: {diff:.2%}")
                print("   Recommendation: Consider K-Fold Cross Validation, more regularization, or more data augmentation.")
            else:
                print(f"✅ {row['Model']} appears healthy (Gap: {diff:.2%})")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
