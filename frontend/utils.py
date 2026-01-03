import os
import re
import json
import random
from pathlib import Path
from typing import Dict, Tuple, cast, Optional, List, Any, Union

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
import google.generativeai as genai # type: ignore
import streamlit as st

# Visualization Imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Constants ---
SUPPORTED_MODELS = ("cnn", "resnet", "efficientnet", "mobilenet", "vit")
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# --- Model Definitions ---

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in (32, 64, 128, 256):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

def build_model(num_classes: int, model_type: str) -> nn.Module:
    model_type = model_type.lower()
    # Handle variations in naming
    if "vit" in model_type: model_type = "vit"
    
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_type '{model_type}'.")

    if model_type == "cnn":
        return SimpleCNN(num_classes)

    if "resnet" in model_type:
        # Fix: Use ResNet50 instead of ResNet18/34 to match checkpoint
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features # 2048
        
        # Reconstruct classifier head to match training script
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        return model

    if "efficientnet" in model_type:
        model = models.efficientnet_b0(weights=None)
        # EfficientNet classifier is model.classifier[1]
        classifier = model.classifier
        if isinstance(classifier[-1], nn.Linear):
            in_features = classifier[-1].in_features
            classifier[-1] = nn.Linear(in_features, num_classes)
        return model
        
    if "vit" in model_type:
        model = models.vit_b_16(weights=None)
        if isinstance(model.heads.head, nn.Linear):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        return model

    if "mobilenet" in model_type:
        model = models.mobilenet_v2(weights=None)
        classifier = model.classifier
        if isinstance(classifier[-1], nn.Linear):
            in_features = classifier[-1].in_features
            classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Architecture {model_type} not implemented.")

@st.cache_resource(show_spinner=False)
def load_model_components(model_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, str], transforms.Compose]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Standard Binary Mapping
    idx_to_class = {"0": "Fake", "1": "Real"}
    class_to_idx = {"Fake": 0, "Real": 1}

    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_type = checkpoint.get("model_type", "efficientnet")
        if "class_names" in checkpoint:
            saved_classes = checkpoint["class_names"]
            idx_to_class = {str(i): name for i, name in enumerate(saved_classes)}
    else:
        state_dict = checkpoint
        model_type = "efficientnet"

    # --- Robust Output Size Detection ---
    # Default to 1 (Binary) as requested, unless we find strong evidence otherwise
    num_outputs = 1 
    
    # Check specific keys for known architectures
    if "fc.4.weight" in state_dict: # ResNet Custom Head
        num_outputs = state_dict["fc.4.weight"].shape[0]
    elif "classifier.1.weight" in state_dict: # EfficientNet
        num_outputs = state_dict["classifier.1.weight"].shape[0]
    elif "heads.head.weight" in state_dict: # ViT
        num_outputs = state_dict["heads.head.weight"].shape[0]
    elif "classifier.5.weight" in state_dict: # SimpleCNN
        num_outputs = state_dict["classifier.5.weight"].shape[0]
    
    try:
        model = build_model(num_outputs, model_type)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        st.warning(f"Architecture mismatch for {model_type}. Attempting strict=False load.")
        try:
            model = build_model(num_outputs, model_type)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e2:
             st.error(f"Critical Error loading {model_type}: {e2}")
             raise e2

    model.to(device).eval()

    image_size = 224
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    return model, idx_to_class, transform

def get_available_models(base_dir: Path) -> Dict[str, Path]:
    models_dir = base_dir / "models"
    model_files = []
    if models_dir.exists():
        model_files.extend(models_dir.rglob("*.pth"))
    model_files.extend(base_dir.glob("*.pth"))
    
    return {f"{f.parent.name}/{f.name}" if f.parent.name != base_dir.name else f.name: f for f in model_files}

def get_random_test_image(base_dir: Path) -> Optional[Tuple[Path, str]]:
    test_dir = base_dir / "data" / "Dataset" / "Test"
    if not test_dir.exists():
        return None
    
    classes = ["Fake", "Real"]
    selected_class = random.choice(classes)
    class_dir = test_dir / selected_class
    
    if not class_dir.exists():
        return None
        
    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
    if not images:
        return None
        
    return random.choice(images), selected_class

# --- Inference Engines ---

def run_gemini_analysis(image: Image.Image) -> Dict:
    if not os.environ.get("GOOGLE_API_KEY"):
        return {"label": "Error", "explanation": "API Key missing."}

    try:
        model = genai.GenerativeModel("gemini-2.5-flash") # type: ignore
        prompt = """
        Analyze this image for Digital Forensics. Detect potential Deepfake manipulation.
        Focus on: Eyes (pupils, reflections), Hands/Fingers, Background consistency, Skin texture artifacts.
        
        Return ONLY valid JSON:
        {
            "verdict": "REAL" or "FAKE",
            "confidence": 0.0 to 1.0,
            "explanation": "Concise technical findings."
        }
        """
        response = model.generate_content([prompt, image])
        
        json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            raise ValueError("Invalid JSON from Gemini")

        verdict = result.get("verdict", "UNKNOWN").upper()
        confidence = float(result.get("confidence", 0.5))
        
        probs = {
            "Fake": confidence if verdict == "FAKE" else 1 - confidence,
            "Real": confidence if verdict == "REAL" else 1 - confidence
        }
        
        return {
            "label": "Fake" if verdict == "FAKE" else "Real",
            "confidence": confidence,
            "probabilities": probs,
            "explanation": result.get("explanation", "N/A")
        }
    except Exception as e:
        return {"label": "Error", "confidence": 0, "probabilities": {}, "explanation": str(e)}

def run_inference_local(image: Image.Image, model: nn.Module, transform: transforms.Compose, device: torch.device, idx_to_class: Dict[str, str]) -> Dict:
    img_rgb = image.convert("RGB")
    image_tensor = cast(torch.Tensor, transform(img_rgb)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        
        if output.shape[1] == 1:
            prob_real = torch.sigmoid(output).item()
            probs = {
                idx_to_class.get("0", "Fake"): 1 - prob_real,
                idx_to_class.get("1", "Real"): prob_real
            }
            pred_idx = 1 if prob_real > 0.5 else 0
            confidence = prob_real if pred_idx == 1 else 1 - prob_real
        else:
            probabilities = torch.softmax(output, dim=1)[0]
            probs = {idx_to_class.get(str(i), str(i)): p.item() for i, p in enumerate(probabilities)}
            confidence, pred_tensor = torch.max(probabilities, dim=0)
            confidence = confidence.item()
            pred_idx = pred_tensor.item()

    predicted_label = idx_to_class.get(str(pred_idx), f"Class {pred_idx}")
    
    cam_viz = None
    try:
        target_layers = None
        if isinstance(model, models.EfficientNet):
            target_layers = [model.features[-1]] # type: ignore
        elif isinstance(model, models.ResNet):
            target_layers = [model.layer4[-1]] # type: ignore
        elif hasattr(model, 'features'):
             target_layers = [model.features[-1]] # type: ignore

        if target_layers:
            cam = GradCAM(model=model, target_layers=target_layers)
            target = [BinaryClassifierOutputTarget(1 if pred_idx == 1 else 0)] if output.shape[1] == 1 else [ClassifierOutputTarget(pred_idx)]
            
            grayscale_cam = cam(input_tensor=image_tensor, targets=cast(Any, target))[0, :]
            
            img_resized = np.array(img_rgb.resize((224, 224))) / 255.0
            cam_viz = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
    except Exception as e:
        print(f"Grad-CAM failed: {e}")

    return {
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": probs,
        "gradcam": cam_viz
    }
