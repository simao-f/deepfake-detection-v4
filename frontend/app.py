import json
import os
import re
import random
from pathlib import Path
from typing import Dict, Tuple, cast, Optional, List
import io

import streamlit as st
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch import nn
from torchvision import models, transforms
import google.generativeai as genai # type: ignore
from dotenv import load_dotenv

# Visualization Imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="DeepFake Sentry",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
gemini_env = Path(__file__).parent.parent / "models" / "gemini" / ".env"
if gemini_env.exists():
    load_dotenv(gemini_env)

# Configure Gemini
if os.environ.get("GOOGLE_API_KEY"):
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY")) # type: ignore

# Constants
SUPPORTED_MODELS = ("cnn", "resnet", "efficientnet", "mobilenet")
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    /* Highlight the prediction box */
    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .fake { background-color: rgba(255, 75, 75, 0.15); border: 2px solid #ff4b4b; color: #ff4b4b; }
    .real { background-color: rgba(0, 200, 83, 0.15); border: 2px solid #00c853; color: #00c853; }
    
    /* Game Styles */
    .game-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MODEL DEFINITIONS ---

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
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_type '{model_type}'.")

    if model_type == "cnn":
        return SimpleCNN(num_classes)

    backbones = {
        "resnet": models.resnet18,
        "mobilenet": models.mobilenet_v2,
        "efficientnet": models.efficientnet_b0,
    }
    model = backbones[model_type](weights=None)

    if model_type == "resnet":
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        return model

    # For EfficientNet/MobileNet
    classifier = model.classifier
    last_layer = classifier[-1]
    if not isinstance(last_layer, nn.Linear):
        in_features = last_layer.in_features
    else:
        in_features = last_layer.in_features
        
    classifier[-1] = nn.Linear(in_features, num_classes)
    return model

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

    num_outputs = len(class_to_idx) 
    if "classifier.1.weight" in state_dict:
        num_outputs = state_dict["classifier.1.weight"].shape[0]
    elif "classifier.1.bias" in state_dict:
        num_outputs = state_dict["classifier.1.bias"].shape[0]
    elif "fc.3.weight" in state_dict:
         num_outputs = state_dict["fc.3.weight"].shape[0]

    try:
        model = build_model(num_outputs, model_type)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        st.warning(f"Architecture mismatch for {model_type}. Attempting strict=False load.")
        model = build_model(num_outputs, model_type)
        model.load_state_dict(state_dict, strict=False)

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

# --- 4. INFERENCE ENGINES ---

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

# --- 5. GAME LOGIC ---

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

# --- 6. UI COMPONENTS ---

def render_analyzer(base_dir: Path, model: Optional[nn.Module], transform: Optional[transforms.Compose], device: torch.device, idx_to_class: Optional[Dict[str, str]], is_gemini: bool):
    st.markdown("### 🕵️ Forensic Image Analysis")
    st.markdown("Upload an image to detect potential deepfake manipulation using our advanced AI models.")

    col_upload, col_demo = st.columns([2, 1])
    
    uploaded_file = None
    with col_upload:
        uploaded_file = st.file_uploader("Upload Suspect Image", type=["png", "jpg", "jpeg", "webp"])
    
    with col_demo:
        st.write("Quick Actions:")
        if st.button("Load Random Test Image"):
            img_data = get_random_test_image(base_dir)
            if img_data:
                img_path, label = img_data
                with open(img_path, "rb") as f:
                    uploaded_file = io.BytesIO(f.read())
                    uploaded_file.name = img_path.name
                st.toast(f"Loaded a random image (Hidden Label)")
                # We need to manually set this in session state to persist if needed, 
                # but for file uploader, we can't easily inject. 
                # Instead, we'll just display it if no file is uploaded but button clicked?
                # Actually, streamlit file_uploader is hard to programmatically set.
                # Let's just return the image object if button clicked.
                st.session_state.analyzer_image = Image.open(img_path)
                st.session_state.analyzer_filename = img_path.name
            else:
                st.error("Could not find test images.")

    # Logic to handle either uploaded file or random loaded image
    image = None
    filename = "Unknown"
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            filename = uploaded_file.name
            # Clear session state if new upload
            if 'analyzer_image' in st.session_state:
                del st.session_state.analyzer_image
        except:
            st.error("Invalid image file.")
            return
    elif 'analyzer_image' in st.session_state:
        image = st.session_state.analyzer_image
        filename = st.session_state.analyzer_filename

    if image:
        c1, c2 = st.columns([1, 1.2])
        
        with c1:
            st.image(image, caption="Suspect Image", use_container_width=True)
            with st.expander("File Metadata"):
                st.json({
                    "Filename": filename,
                    "Dimensions": f"{image.size}",
                    "Mode": image.mode
                })

        with c2:
            st.markdown("### Analysis Console")
            analyze_btn = st.button("🔍 Run Forensic Analysis", type="primary", use_container_width=True)
            
            if analyze_btn:
                with st.spinner("Processing neural pathways..."):
                    result = None
                    if is_gemini:
                        result = run_gemini_analysis(image)
                    elif model and transform and idx_to_class:
                        result = run_inference_local(image, model, transform, device, idx_to_class)
                    else:
                        st.error("No model loaded. Please select a model from the sidebar.")
                        return
                    
                    if result and result.get("label") != "Error":
                        label = str(result.get("label", "Unknown"))
                        conf = result["confidence"]
                        
                        color_class = "fake" if label.lower() == "fake" else "real"
                        icon = "🚨" if label.lower() == "fake" else "✅"
                        
                        st.markdown(f"""
                        <div class="prediction-box {color_class}">
                            <h2>{icon} VERDICT: {label.upper()}</h2>
                            <p>Confidence Level</p>
                            <h1>{conf:.1%}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        tab1, tab2, tab3 = st.tabs(["📊 Metrics", "🧠 Explainability", "📥 Export"])
                        
                        with tab1:
                            st.bar_chart(result["probabilities"], color="#ff4b4b" if label == "Fake" else "#00c853")
                            if "explanation" in result:
                                st.info(result["explanation"])
                                
                        with tab2:
                            if result.get("gradcam") is not None:
                                st.write("**Class Activation Map (Heatmap)**")
                                alpha = st.slider("Heatmap Intensity", 0.0, 1.0, 0.6)
                                
                                cam_img = Image.fromarray(result["gradcam"])
                                orig_resized = image.convert("RGB").resize((224, 224))
                                blended = Image.blend(orig_resized, cam_img, alpha)
                                
                                st.image(blended, caption="Red areas influenced prediction", use_container_width=True)
                            else:
                                st.caption("Explainability not available for this model type.")
                                
                        with tab3:
                            report_json = json.dumps(result, indent=4, default=str)
                            st.download_button(
                                label="Download Forensic Report (JSON)",
                                data=report_json,
                                file_name="forensic_report.json",
                                mime="application/json"
                            )
                    else:
                        st.error(f"Analysis Failed: {result.get('explanation', 'Unknown Error')}")

def render_game(base_dir: Path):
    st.markdown("### 🎓 Training Mode: Human vs AI")
    st.markdown("Test your own deepfake detection skills! Can you spot the fake?")

    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'total' not in st.session_state:
        st.session_state.total = 0
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'current_label' not in st.session_state:
        st.session_state.current_label = None
    if 'game_state' not in st.session_state:
        st.session_state.game_state = "guessing" # guessing, result

    col_score, col_action = st.columns([1, 3])
    
    with col_score:
        st.metric("Your Score", f"{st.session_state.score}/{st.session_state.total}")
        if st.button("Reset Score"):
            st.session_state.score = 0
            st.session_state.total = 0
            st.rerun()

    with col_action:
        if st.session_state.current_image is None:
             if st.button("Start Challenge"):
                load_new_game_image(base_dir)
        elif st.button("Skip / Load New"):
             load_new_game_image(base_dir)

    if st.session_state.current_image:
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            image = Image.open(st.session_state.current_image)
            st.image(image, caption="Is this Real or Fake?", use_container_width=True)
        
        with c2:
            st.markdown("### Make your guess:")
            
            if st.session_state.game_state == "guessing":
                col_real, col_fake = st.columns(2)
                with col_real:
                    if st.button("✅ REAL", use_container_width=True):
                        check_guess("Real")
                        st.rerun()
                with col_fake:
                    if st.button("🚨 FAKE", use_container_width=True):
                        check_guess("Fake")
                        st.rerun()
            else:
                correct = st.session_state.last_guess == st.session_state.current_label
                if correct:
                    st.success(f"🎉 Correct! It was {st.session_state.current_label.upper()}.")
                else:
                    st.error(f"❌ Wrong! It was actually {st.session_state.current_label.upper()}.")
                
                if st.button("Next Image ➡️", type="primary", use_container_width=True):
                    load_new_game_image(base_dir)

def load_new_game_image(base_dir):
    img_data = get_random_test_image(base_dir)
    if img_data:
        st.session_state.current_image = img_data[0]
        st.session_state.current_label = img_data[1]
        st.session_state.game_state = "guessing"
        st.rerun()
    else:
        st.error("Could not load images from dataset.")

def check_guess(guess):
    st.session_state.last_guess = guess
    st.session_state.total += 1
    if guess == st.session_state.current_label:
        st.session_state.score += 1
    st.session_state.game_state = "result"

# --- 7. MAIN ENTRY POINT ---

def main():
    base_dir = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🛡️ Sentry Control")
        
        mode = st.radio("Select Mode", ["Forensic Analyzer", "Training Game"], index=0)
        
        st.divider()
        
        # Hardware Status
        if torch.cuda.is_available():
            st.success(f"🚀 GPU Active: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            st.info("🍎 Apple Silicon (MPS) Active")
            device = torch.device("mps")
        else:
            st.warning("🐌 CPU Mode Active")
            device = torch.device("cpu")

        st.divider()
        
        # Model Selection (Only for Analyzer)
        model = None
        transform = None
        idx_to_class = None
        is_gemini = False
        
        if mode == "Forensic Analyzer":
            st.subheader("Model Settings")
            available_models = get_available_models(base_dir)
            model_options = ["Gemini 2.5 Flash (Cloud)"] + list(available_models.keys())
            
            selected_model_name = st.selectbox("Intelligence Model", model_options)
            is_gemini = "Gemini" in selected_model_name
            
            if not is_gemini:
                model_path = available_models[selected_model_name]
                try:
                    model, idx_to_class, transform = load_model_components(model_path, device)
                    st.caption(f"Loaded: {model_path.name}")
                except Exception as e:
                    st.error(f"Error loading model: {e}")

    # --- ROUTING ---
    if mode == "Forensic Analyzer":
        render_analyzer(base_dir, model, transform, device, idx_to_class, is_gemini)
    else:
        render_game(base_dir)

if __name__ == "__main__":
    main()
