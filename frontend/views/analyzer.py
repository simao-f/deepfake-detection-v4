import json
import io
from pathlib import Path
from typing import Dict, Optional, Any

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from frontend.utils import run_gemini_analysis, run_inference_local, get_random_test_image

def render_analyzer(
    base_dir: Path, 
    model: Optional[nn.Module], 
    transform: Optional[transforms.Compose], 
    device: torch.device, 
    idx_to_class: Optional[Dict[str, str]], 
    is_gemini: bool,
    all_models: Optional[Dict[str, Any]] = None
):
    st.markdown("### Deepfake Detection Analysis")
    st.markdown("Upload an image to analyze for deepfake manipulation.")

    col_upload, col_demo = st.columns([2, 1])
    
    # --- Image Loading Logic ---
    if 'analyzer_image' not in st.session_state:
        st.session_state.analyzer_image = None
        st.session_state.analyzer_filename = None
    
    uploaded_file = None
    with col_upload:
        uploaded_file = st.file_uploader("Upload Suspect Image", type=["png", "jpg", "jpeg", "webp"])
    
    with col_demo:
        st.write("Quick Actions:")
        if st.button("Load Random Test Image"):
            img_data = get_random_test_image(base_dir)
            if img_data:
                img_path, label = img_data
                st.toast(f"Loaded a random image (Hidden Label)")
                st.session_state.analyzer_image = Image.open(img_path)
                st.session_state.analyzer_filename = img_path.name
                st.rerun()
            else:
                st.error("Could not find test images.")

    # Determine which image to show
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            filename = uploaded_file.name
            st.session_state.analyzer_image = image
            st.session_state.analyzer_filename = filename
        except:
            st.error("Invalid image file.")
            return
    
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
            analyze_btn = st.button("Run Analysis", type="primary", use_container_width=True)
            
            if analyze_btn:
                with st.spinner("Analyzing image..."):
                    
                    # --- ALL MODELS MODE ---
                    if all_models:
                        st.subheader("Ensemble Analysis Results")
                        results = {}
                        
                        # Run local models
                        for name, (m, t, i) in all_models.items():
                            res = run_inference_local(image, m, t, device, i)
                            results[name] = res
                        
                        # Display results
                        for name, res in results.items():
                            label = str(res.get("label", "Unknown"))
                            conf = res["confidence"]
                            color = "red" if label.lower() == "fake" else "green"
                            st.markdown(f"**{name}**: :{color}[{label}] ({conf:.1%})")
                            
                        # Aggregate Verdict
                        fake_votes = sum(1 for r in results.values() if r.get("label", "").lower() == "fake")
                        total_votes = len(results)
                        final_verdict = "FAKE" if fake_votes > total_votes / 2 else "REAL"
                        
                        st.divider()
                        st.markdown(f"### Ensemble Verdict: **{final_verdict}**")
                        
                    # --- SINGLE MODEL MODE ---
                    else:
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
                            
                            st.markdown(f"""
                            <div class="prediction-box {color_class}">
                                <h2>VERDICT: {label.upper()}</h2>
                                <p>Confidence Level</p>
                                <h1>{conf:.1%}</h1>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            tab1, tab2, tab3 = st.tabs(["Metrics", "Explainability", "Export"])
                            
                            with tab1:
                                st.bar_chart(result["probabilities"], color="#ff4b4b" if label == "Fake" else "#00c853")
                                if "explanation" in result:
                                    st.info(result["explanation"])
                                    
                            with tab2:
                                if result.get("gradcam") is not None:
                                    st.write("**Class Activation Map (Heatmap)**")
                                    cam_img = Image.fromarray(result["gradcam"])
                                    orig_resized = image.convert("RGB").resize((224, 224))
                                    blended = Image.blend(orig_resized, cam_img, 0.6)
                                    st.image(blended, caption="Red areas influenced prediction", use_container_width=True)
                                else:
                                    st.caption("Explainability not available for this model type.")
                                    
                            with tab3:
                                report_json = json.dumps(result, indent=4, default=str)
                                st.download_button(
                                    label="Download Report (JSON)",
                                    data=report_json,
                                    file_name="forensic_report.json",
                                    mime="application/json"
                                )
                        else:
                            st.error(f"Analysis Failed: {result.get('explanation', 'Unknown Error')}")
