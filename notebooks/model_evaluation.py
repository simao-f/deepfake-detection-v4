import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 256
NUM_WORKERS = 8
IMG_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["Fake", "Real"]

# Default Paths
DEFAULT_PATHS = {
    "efficientnet": "models/efficientnet/efficientnet_b0_deepfake_best.pth",
    "resnet": "models/resnet/resnet50_deepfake.pth",
    "vit": "models/vision_transformer/vit_deepfake_detection.pth"
}

DATASETS = {
    "Train": "data/Dataset/Train",
    "Validation": "data/Dataset/Validation",
    "Test": "data/Dataset/Test"
}

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def build_model(architecture: str, num_classes: int = 1) -> nn.Module:
    architecture = architecture.lower()
    
    if architecture == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        classifier = model.classifier
        if isinstance(classifier[1], nn.Linear):
            in_features = classifier[1].in_features
        else:
            in_features = 1280
            
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1) # Always 1 for EfficientNet in this project
        )
        
    elif architecture == "resnet":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential( # type: ignore
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1) # Always 1 for ResNet in this project
        )
        
    elif architecture == "vit":
        model = models.vit_b_16(weights=None)
        heads = model.heads 
        if isinstance(heads.head, nn.Linear):
            in_features = heads.head.in_features
        else:
            in_features = 768
            
        model.heads.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 2) # Always 2 for ViT in this project
        )
        
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model

def load_model(architecture: str, weights_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    logger.info(f"Loading {architecture} model from {weights_path}...")
    
    checkpoint = torch.load(weights_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    last_layer_keys = [k for k in clean_state_dict.keys() if "classifier" in k or "fc" in k or "head" in k]
    last_weight_key = next((k for k in reversed(last_layer_keys) if "weight" in k), None)
    
    num_classes = 1
    if last_weight_key:
        output_dim = clean_state_dict[last_weight_key].shape[0]
        num_classes = output_dim

    model = build_model(architecture, num_classes)
    
    try:
        model.load_state_dict(clean_state_dict)
    except RuntimeError as e:
        logger.warning(f"Strict loading failed: {e}. Retrying with strict=False.")
        model.load_state_dict(clean_state_dict, strict=False)

    model.to(device)
    model.eval()
    return model

def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_index]
    eer_threshold = thresholds[eer_index]
    return float(eer), float(eer_threshold)

def run_inference(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = []
    y_scores = []
    y_preds = []

    use_amp = device.type == "cuda"
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'): # type: ignore
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs > 0.5).long()
                scores = probs
            else:
                probs = torch.softmax(outputs, dim=1)
                scores = probs[:, 1]
                _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_scores.extend(scores.float().cpu().numpy())
            y_preds.extend(preds.cpu().numpy())

    return np.array(y_true), np.array(y_scores), np.array(y_preds)

def evaluate_dataset(model: nn.Module, data_dir: str, device: torch.device) -> Dict[str, float]:
    if not os.path.exists(data_dir):
        logger.error(f"Dataset path not found: {data_dir}")
        return {}

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    y_true, y_scores, y_preds = run_inference(model, dataloader, device)

    # Calculate metrics and cast to native float to avoid numpy types in Dict return
    acc = float(accuracy_score(y_true, y_preds))
    prec = float(precision_score(y_true, y_preds, zero_division=0)) 
    rec = float(recall_score(y_true, y_preds, zero_division=0)) 
    f1 = float(f1_score(y_true, y_preds, zero_division=0)) 
    
    auc = 0.0
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        auc = 0.0 
        
    eer, eer_thresh = calculate_eer(y_true, y_scores)

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "AUC-ROC": auc,
        "EER": eer,
        "EER Threshold": eer_thresh
    }

def print_markdown_table(df: pd.DataFrame, title: str):
    print(f"\n### {title}")
    # Ensure tabulate is installed or fall back to string
    try:
        print(df.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        print(df.to_string(index=False))
    print("\n")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Deepfake Model Evaluation")
    parser.add_argument("--output_dir", type=str, default="logs/plots", help="Directory to save results")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Initialize storage for results
    all_results: Dict[str, List[Dict[str, Union[str, float]]]] = {phase: [] for phase in DATASETS.keys()}

    # Loop through Models
    for model_name, weights_path in DEFAULT_PATHS.items():
        logger.info(f"Processing Model: {model_name.upper()}")
        
        try:
            model = load_model(model_name, weights_path, device)
        except Exception as e:
            logger.error(f"Skipping {model_name}: {e}")
            continue

        # Loop through Datasets (Phases)
        for phase, data_path in DATASETS.items():
            logger.info(f"  Evaluating on {phase} set...")
            metrics = evaluate_dataset(model, data_path, device)
            
            if metrics:
                record = {"Model": model_name.capitalize(), **metrics}
                all_results[phase].append(record)

    # Output Tables
    print("\n" + "="*60)
    print("  COMPREHENSIVE EVALUATION REPORT")
    print("="*60)

    for phase in ["Train", "Validation", "Test"]:
        if all_results[phase]:
            df = pd.DataFrame(all_results[phase])
            # Reorder columns
            cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "EER", "EER Threshold"]
            # Filter columns to ensure they exist
            existing_cols = [c for c in cols if c in df.columns]
            df = df[existing_cols]
            
            print_markdown_table(df, f"{phase} Set Performance")
            
            # Save to CSV
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path / f"evaluation_report_{phase.lower()}.csv", index=False)

if __name__ == "__main__":
    main()