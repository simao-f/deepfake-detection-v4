import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ODD"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = LOGS_DIR / "ood_test_results.txt"
OUTPUT_CM_FILE = LOGS_DIR / "ood_confusion_matrices.png"

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions (Must match training) ---
def build_model(model_name: str) -> nn.Module:
    model_name = model_name.lower()
    
    if "resnet" in model_name:
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
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
        classifier = model.classifier
        if isinstance(classifier[1], nn.Linear):
            in_features = classifier[1].in_features
        else:
            in_features = 1280
            
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1)
        )
        return model
        
    if "vit" in model_name:
        model = models.vit_b_16(weights=None)
        if isinstance(model.heads.head, nn.Linear):
            in_features = model.heads.head.in_features
        else:
            in_features = 768
            
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
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        print(f"Error: Unknown checkpoint format for {model_name}")
        return None

    # Handle DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
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
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs > 0.5).float()
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    return np.array(all_labels), np.array(all_probs), np.array(all_preds)

def main():
    print(f"--- OOD Testing on {DATA_DIR} ---")
    
    # 1. Setup Data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = datasets.ImageFolder(str(DATA_DIR), transform=transform)
    except FileNotFoundError:
        print(f"Error: OOD Data directory not found at {DATA_DIR}")
        return

    print(f"Found {len(dataset)} images.")
    print(f"Classes: {dataset.classes}")
    
    # Ensure we have exactly 1000 fake and 1000 real if possible, or just use all.
    # The user asked for 1000 fake and 1000 real.
    # Let's check indices.
    targets = np.array(dataset.targets)
    class_to_idx = dataset.class_to_idx
    
    # Assuming 'fake' is 0 and 'real' is 1 (alphabetical)
    fake_idx = class_to_idx.get('fake')
    real_idx = class_to_idx.get('real')
    
    if fake_idx is None or real_idx is None:
        # Fallback if capitalization differs
        fake_idx = class_to_idx.get('Fake') or 0
        real_idx = class_to_idx.get('Real') or 1
        
    fake_indices = np.where(targets == fake_idx)[0]
    real_indices = np.where(targets == real_idx)[0]
    
    print(f"Found {len(fake_indices)} Fake images and {len(real_indices)} Real images.")
    
    # Select 1000 of each
    selected_fake = fake_indices[:1000]
    selected_real = real_indices[:1000]
    
    subset_indices = np.concatenate([selected_fake, selected_real])
    subset_dataset = Subset(dataset, subset_indices)
    
    dataloader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Testing on {len(subset_dataset)} images (1000 Fake, 1000 Real).")

    # 2. Define Models
    models_to_eval = [
        {
            "name": "EfficientNet",
            "path": MODELS_DIR / "efficientnet" / "efficientnet_b0_deepfake_best.pth",
            "type": "binary"
        },
        {
            "name": "ResNet50",
            "path": MODELS_DIR / "resnet" / "resnet50_deepfake.pth",
            "type": "binary"
        },
        {
            "name": "ViT",
            "path": MODELS_DIR / "vision_transformer" / "vit_deepfake_detection.pth",
            "type": "multiclass"
        }
    ]
    
    results = {}
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("--- OOD Test Results (Dataset: ODD) ---\n\n")
        
        for m in models_to_eval:
            print(f"\nEvaluating {m['name']}...")
            model = load_model(m['name'], m['path'], DEVICE)
            
            if model is None:
                continue
                
            labels, probs, preds = evaluate_model(model, dataloader, DEVICE, m['type'])
            
            acc = accuracy_score(labels, preds)
            auc = roc_auc_score(labels, probs)
            
            report = f"{m['name']}:\n"
            report += f"  Accuracy: {acc:.4f}\n"
            report += f"  AUC:      {auc:.4f}\n"
            report += "  Confusion Matrix:\n"
            cm = confusion_matrix(labels, preds)
            report += str(cm) + "\n"
            report += "  Classification Report:\n"
            report += classification_report(labels, preds, target_names=['Fake', 'Real'])
            report += "\n" + "-"*30 + "\n"
            
            print(report)
            f.write(report)
            
            results[m['name']] = {
                "labels": labels,
                "preds": preds,
                "acc": acc
            }

    # Plot Confusion Matrices
    if results:
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
        if len(results) == 1: axes = [axes]
        
        for ax, (name, res) in zip(axes, results.items()):
            cm = confusion_matrix(res['labels'], res['preds'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
            ax.set_title(f"{name} (OOD)\nAcc: {res['acc']:.2%}")
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            
        plt.tight_layout()
        plt.savefig(OUTPUT_CM_FILE)
        print(f"Confusion Matrices saved to {OUTPUT_CM_FILE}")

if __name__ == "__main__":
    main()
