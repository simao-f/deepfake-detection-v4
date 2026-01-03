import os
import sys
from pathlib import Path
import streamlit as st
import torch
import google.generativeai as genai # type: ignore
from dotenv import load_dotenv

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import modularized components
from frontend.utils import load_model_components, get_available_models
from frontend.views.analyzer import render_analyzer
from frontend.views.game import render_game

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="DeepFake Sentry",
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

# --- 3. MAIN ENTRY POINT ---

def main():
    base_dir = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("Sentry Control")
        
        mode = st.radio("Select Mode", ["Deepfake Detector", "Deepfake Game"], index=0)
        
        st.divider()
        
        # Hardware Status
        if torch.cuda.is_available():
            st.success(f"GPU Active: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            st.info("Apple Silicon (MPS) Active")
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        st.divider()
        
        # Model Selection
        model = None
        transform = None
        idx_to_class = None
        is_gemini = False
        ai_models = {}
        all_models_dict = None # For Analyzer Ensemble
        
        if mode == "Deepfake Detector":
            st.subheader("Model Settings")
            available_models = get_available_models(base_dir)
            model_options = ["All Models (Ensemble)", "Gemini 2.5 Flash (Cloud)"] + list(available_models.keys())
            
            selected_model_name = st.selectbox("Intelligence Model", model_options)
            is_gemini = "Gemini" in selected_model_name
            is_ensemble = "All Models" in selected_model_name
            
            if is_ensemble:
                with st.spinner("Loading All Models..."):
                    all_models_dict = {}
                    for name, path in available_models.items():
                        try:
                            m, i, t = load_model_components(path, device)
                            all_models_dict[name] = (m, t, i)
                        except Exception as e:
                            st.error(f"Failed to load {name}: {e}")
                st.success(f"Loaded {len(all_models_dict)} models for ensemble analysis.")
            
            elif not is_gemini:
                model_path = available_models[selected_model_name]
                try:
                    model, idx_to_class, transform = load_model_components(model_path, device)
                    st.caption(f"Loaded: {model_path.name}")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
        
        elif mode == "Deepfake Game":
            st.subheader("AI Opponents")
            available_models = get_available_models(base_dir)
            selected_opponents = st.multiselect(
                "Select Models to Compete Against", 
                list(available_models.keys()),
                default=list(available_models.keys())
            )
            
            if selected_opponents:
                with st.spinner("Loading Opponents..."):
                    for name in selected_opponents:
                        path = available_models[name]
                        try:
                            m, i, t = load_model_components(path, device)
                            ai_models[name] = (m, t, i)
                        except Exception as e:
                            st.error(f"Failed to load {name}: {e}")
                st.success(f"Loaded {len(ai_models)} opponents.")

    # --- ROUTING ---
    if mode == "Deepfake Detector":
        render_analyzer(base_dir, model, transform, device, idx_to_class, is_gemini, all_models_dict)
    else:
        render_game(base_dir, ai_models, device)

if __name__ == "__main__":
    main()
