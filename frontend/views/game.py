import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
import torch
from PIL import Image

from frontend.utils import run_inference_local, get_random_test_image

HISTORY_FILE = Path("logs/game_history.csv")

def save_game_result(filename: str, true_label: str, user_guess: str, ai_results: Dict[str, str]):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = HISTORY_FILE.exists()
    
    with open(HISTORY_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Image", "Ground Truth", "User Guess", "User Correct", "AI Predictions"])
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_correct = user_guess == true_label
        
        writer.writerow([
            timestamp, 
            filename, 
            true_label, 
            user_guess, 
            user_correct, 
            json.dumps(ai_results)
        ])

def load_game_history() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(HISTORY_FILE)
    # Parse AI Predictions JSON
    def parse_ai(x):
        try:
            return json.loads(x)
        except:
            return {}
            
    df["AI Predictions"] = df["AI Predictions"].apply(parse_ai)
    return df

def process_guess(user_guess: str, ai_models: Dict[str, Any], device: torch.device):
    # Run AI Inference
    ai_results = {}
    if ai_models and st.session_state.current_image:
        try:
            image = Image.open(st.session_state.current_image)
            for name, (model, transform, idx_to_class) in ai_models.items():
                res = run_inference_local(image, model, transform, device, idx_to_class)
                ai_results[name] = res["label"]
        except Exception as e:
            print(f"AI Inference failed: {e}")

    # Save Result
    if st.session_state.current_image:
        save_game_result(
            st.session_state.current_image.name, 
            st.session_state.current_label, 
            user_guess, 
            ai_results
        )
    
    # Update Session State
    st.session_state.last_guess = user_guess
    st.session_state.last_ai_results = ai_results
    st.session_state.total += 1
    if user_guess == st.session_state.current_label:
        st.session_state.score += 1
    st.session_state.game_state = "result"

def load_new_game_image(base_dir: Path):
    img_data = get_random_test_image(base_dir)
    if img_data:
        st.session_state.current_image = img_data[0]
        st.session_state.current_label = img_data[1]
        st.session_state.game_state = "guessing"
        st.session_state.last_ai_results = {}
        st.rerun()
    else:
        st.error("Could not load images from dataset.")

def render_game(base_dir: Path, ai_models: Dict[str, Any], device: torch.device):
    st.markdown("### Deepfake Game")
    st.markdown("Test your detection skills against the AI models.")

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
    if 'last_ai_results' not in st.session_state:
        st.session_state.last_ai_results = {}

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
                    if st.button("REAL", use_container_width=True):
                        process_guess("Real", ai_models, device)
                        st.rerun()
                with col_fake:
                    if st.button("FAKE", use_container_width=True):
                        process_guess("Fake", ai_models, device)
                        st.rerun()
            else:
                correct = st.session_state.last_guess == st.session_state.current_label
                if correct:
                    st.success(f"Result: Correct. It was {st.session_state.current_label.upper()}.")
                else:
                    st.error(f"Result: Incorrect. It was actually {st.session_state.current_label.upper()}.")
                
                # Show AI Results
                if st.session_state.last_ai_results:
                    st.markdown("#### AI Model Predictions:")
                    for name, pred in st.session_state.last_ai_results.items():
                        ai_correct = pred == st.session_state.current_label
                        status = "Correct" if ai_correct else "Incorrect"
                        st.write(f"**{name}**: Guessed {pred} ({status})")
                
                if st.button("Next Image", type="primary", use_container_width=True):
                    load_new_game_image(base_dir)

    # --- History & Analysis Section ---
    st.divider()
    with st.expander("Session History", expanded=False):
        df = load_game_history()
        if not df.empty:
            # 1. Stats
            total_games = len(df)
            user_acc = df["User Correct"].mean()
            
            st.markdown(f"**Total Games:** {total_games} | **Your Accuracy:** {user_acc:.1%}")
            
            # AI Stats
            ai_stats = {}
            
            # Expand AI Predictions into columns
            ai_preds_df = pd.json_normalize(df["AI Predictions"])
            if not ai_preds_df.empty:
                combined = pd.concat([df.reset_index(drop=True), ai_preds_df], axis=1)
                
                # Calculate AI Accuracy
                for col in ai_preds_df.columns:
                    # Filter where model made a prediction (not NaN)
                    valid_preds = combined[combined[col].notna()]
                    if not valid_preds.empty:
                        acc = (valid_preds[col] == valid_preds["Ground Truth"]).mean()
                        ai_stats[col] = acc
                
                if ai_stats:
                    st.markdown("#### AI Accuracy Comparison")
                    stats_df = pd.DataFrame({
                        "Player": ["You"] + list(ai_stats.keys()),
                        "Accuracy": [user_acc] + list(ai_stats.values())
                    })
                    st.bar_chart(stats_df.set_index("Player"))
            
            # 2. Data Table
            st.dataframe(df[["Timestamp", "Image", "Ground Truth", "User Guess", "User Correct", "AI Predictions"]].sort_values("Timestamp", ascending=False))
        else:
            st.info("No game history yet. Play a round!")
