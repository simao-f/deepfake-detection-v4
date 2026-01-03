import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Dataset" / "Test"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

OUTPUT_ROC_FILE = LOGS_DIR / "roc_curves_comparison.png"
OUTPUT_CM_FILE = LOGS_DIR / "confusion_matrices.png"

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions ---
def build_model(model_name: str) -> nn.Module:
    model_name = model_name.lower()
    
    if "resnet" in model_name:
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        # Match training architecture: Linear -> BN -> ReLU -> Dropout(0.5) -> Linear(1)
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        return model

    if "efficientnet" in model_name:
        model = models.efficientnet_b0(weights=None)
        # Match training architecture: Dropout(0.5) -> Linear(1)
        classifier = model.classifier
        if isinstance(classifier[1], nn.Linear):
            in_features = classifier[1].in_features
        else:
            in_features = 1280 # Default for B0
            
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1)
        )
        return model
        
    if "vit" in model_name:
        model = models.vit_b_16(weights=None)
        # Match training architecture: Dropout(0.5) -> Linear(2)
        if isinstance(model.heads.head, nn.Linear):
            in_features = model.heads.head.in_features
        else:
            in_features = 768 # Default for ViT-B/16
            
        model.heads.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 2)
        )
        return model

    raise ValueError(f"Architecture {model_name} not implemented.")

def load_model(model_name: str, model_path: Path, device: torch.device) -> nn.Module:
    print(f"Loading {model_name} from {model_path}...")
    model = build_model(model_name)
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different saving formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        print(f"Error: Unknown checkpoint format for {model_name}")
        return None

    # Handle DataParallel (remove 'module.' prefix)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # Load weights
    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"Error loading state dict for {model_name}: {e}")
        return None
        
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, device, model_type="binary"):
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            if model_type == "binary":
                # Output shape (N, 1) -> Sigmoid
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs > 0.5).float()
            else:
                # Output shape (N, 2) -> Softmax
                probs = torch.softmax(outputs, dim=1)[:, 1] # Prob of class 1 (Fake)
                _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    return np.array(all_labels), np.array(all_probs), np.array(all_preds)

def main():
    # 1. Setup Data
    print(f"Looking for Test data in {DATA_DIR}")
    if not DATA_DIR.exists():
        print("Test directory not found!")
        return

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(str(DATA_DIR), transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Found {len(dataset)} test images.")
    print(f"Classes: {dataset.classes}") # Should be ['Fake', 'Real'] or similar. 
    # Note: ImageFolder sorts classes alphabetically. 
    # If Fake=0, Real=1, then class 1 is Real. 
    # If Fake is target, we need to be careful. 
    # Usually 'Fake' comes before 'Real', so Fake=0, Real=1.
    # But typically we want to detect Fakes. 
    # Let's assume the models were trained with whatever ImageFolder assigned.
    # If trained with Fake=0, Real=1, then "Positive" usually means 1.
    # We will plot ROC for class 1.
    
    # 2. Define Models to Evaluate
    models_to_eval = [
        {
            "name": "EfficientNet",
            "path": MODELS_DIR / "efficientnet" / "efficientnet_b0_deepfake_best.pth",
            "type": "binary" # 1 output
        },
        {
            "name": "ResNet50",
            "path": MODELS_DIR / "resnet" / "resnet50_deepfake.pth",
            "type": "binary" # 1 output
        },
        {
            "name": "ViT",
            "path": MODELS_DIR / "vision_transformer" / "vit_deepfake_detection.pth",
            "type": "multiclass" # 2 outputs
        }
    ]
    
    plt.figure(figsize=(10, 8))
    
    results = {}

    for m in models_to_eval:
        print(f"\n--- Evaluating {m['name']} ---")
        model = load_model(m['name'], m['path'], DEVICE)
        
        if model is None:
            print(f"Skipping {m['name']} due to load error.")
            continue
            
        labels, probs, preds = evaluate_model(model, dataloader, DEVICE, m['type'])
        
        # Calculate Metrics
        acc = accuracy_score(labels, preds)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        print(f"{m['name']} Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUC:      {roc_auc:.4f}")
        
        results[m['name']] = {
            "labels": labels,
            "preds": preds,
            "acc": acc,
            "auc": roc_auc
        }
        
        plt.plot(fpr, tpr, lw=2, label=f'{m["name"]} (AUC = {roc_auc:.2f})')

    # 3. Plot ROC
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison (Test Set)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(OUTPUT_ROC_FILE)
    print(f"\nROC Curve saved to {OUTPUT_ROC_FILE}")
    
    # 4. Plot Confusion Matrices
    if results:
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
        if len(results) == 1: axes = [axes]
        
        for ax, (name, res) in zip(axes, results.items()):
            cm = confusion_matrix(res['labels'], res['preds'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=dataset.classes, yticklabels=dataset.classes)
            ax.set_title(f"{name}\nAcc: {res['acc']:.2%}")
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            
        plt.tight_layout()
        plt.savefig(OUTPUT_CM_FILE)
        print(f"Confusion Matrices saved to {OUTPUT_CM_FILE}")

if __name__ == "__main__":
    main()
