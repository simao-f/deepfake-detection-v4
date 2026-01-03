import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import ast
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuration
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
HISTORY_FILE = LOGS_DIR / "game_history.csv"
OUTPUT_DIR = LOGS_DIR / "game_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    if not HISTORY_FILE.exists():
        print(f"Error: {HISTORY_FILE} not found.")
        sys.exit(1)
    
    df = pd.read_csv(HISTORY_FILE)
    print(f"Loaded {len(df)} game records.")
    return df

def parse_ai_predictions(df):
    # The AI Predictions column contains JSON strings.
    # We need to parse them and expand them into columns.
    
    def safe_json_loads(x):
        try:
            # Handle potential single quotes or other issues if not valid JSON
            if isinstance(x, dict): return x
            return json.loads(x)
        except:
            try:
                return ast.literal_eval(x)
            except:
                return {}

    ai_preds = df["AI Predictions"].apply(safe_json_loads).apply(pd.Series)
    
    # Rename columns to indicate they are AI predictions
    ai_preds = ai_preds.add_prefix("AI_")
    
    # Combine with original dataframe
    df_combined = pd.concat([df, ai_preds], axis=1)
    return df_combined, ai_preds.columns.tolist()

def analyze_performance(df, ai_cols):
    results = []
    
    # 1. User Performance
    user_acc = accuracy_score(df["Ground Truth"], df["User Guess"])
    results.append({"Player": "User", "Accuracy": user_acc})
    
    # 2. AI Performance
    for col in ai_cols:
        # Filter out NaNs (models might not have run for all games)
        valid_rows = df[df[col].notna()]
        if len(valid_rows) > 0:
            acc = accuracy_score(valid_rows["Ground Truth"], valid_rows[col])
            results.append({"Player": col.replace("AI_", ""), "Accuracy": acc})
            
    return pd.DataFrame(results)

def generate_report(df, performance_df, ai_cols):
    report_path = OUTPUT_DIR / "analysis_report.txt"
    
    with open(report_path, "w") as f:
        f.write("="*50 + "\n")
        f.write("DEEPFAKE GAME ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total Games Played: {len(df)}\n")
        f.write(f"Date Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}\n\n")
        
        f.write("-" * 30 + "\n")
        f.write("LEADERBOARD\n")
        f.write("-" * 30 + "\n")
        f.write(performance_df.sort_values("Accuracy", ascending=False).to_markdown(index=False, floatfmt=".2%"))
        f.write("\n\n")
        
        f.write("-" * 30 + "\n")
        f.write("DETAILED METRICS\n")
        f.write("-" * 30 + "\n")
        
        # User Report
        f.write("USER METRICS:\n")
        f.write(classification_report(df["Ground Truth"], df["User Guess"]))
        f.write("\n")
        
        # AI Reports
        for col in ai_cols:
            model_name = col.replace("AI_", "")
            valid_rows = df[df[col].notna()]
            if len(valid_rows) > 0:
                f.write(f"{model_name.upper()} METRICS:\n")
                f.write(classification_report(valid_rows["Ground Truth"], valid_rows[col]))
                f.write("\n")

        # Hardest Images
        f.write("-" * 30 + "\n")
        f.write("HARDEST IMAGES (User Failed)\n")
        f.write("-" * 30 + "\n")
        failed_games = df[df["User Correct"] == False]
        if not failed_games.empty:
            for _, row in failed_games.iterrows():
                f.write(f"Image: {row['Image']} | Truth: {row['Ground Truth']} | User Guess: {row['User Guess']}\n")
        else:
            f.write("None! Perfect score.\n")

    print(f"Report saved to {report_path}")

def plot_results(performance_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=performance_df, x="Player", y="Accuracy", palette="viridis")
    plt.title("Deepfake Detection Accuracy: Human vs AI")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_comparison.png")
    print(f"Plot saved to {OUTPUT_DIR / 'accuracy_comparison.png'}")

def main():
    print("Starting Game Analysis...")
    df = load_data()
    
    if df.empty:
        print("No data found in game history.")
        return

    df_processed, ai_cols = parse_ai_predictions(df)
    
    # Convert boolean 'User Correct' to proper boolean if it's string
    if df_processed["User Correct"].dtype == object:
        df_processed["User Correct"] = df_processed["User Correct"].map({'True': True, 'False': False, True: True, False: False})

    performance_df = analyze_performance(df_processed, ai_cols)
    
    print("\n--- Performance Summary ---")
    print(performance_df.to_markdown(index=False, floatfmt=".2%"))
    
    generate_report(df_processed, performance_df, ai_cols)
    
    try:
        plot_results(performance_df)
    except Exception as e:
        print(f"Could not generate plots: {e}")

if __name__ == "__main__":
    main()
