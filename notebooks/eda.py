import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import logging

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data/Dataset"
OUTPUT_DIR = PROJECT_ROOT / "logs/eda_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "eda.log")
    ]
)
logger = logging.getLogger(__name__)

def collect_metadata(data_dir: Path) -> pd.DataFrame:
    """
    Traverses the dataset directory and collects metadata for every image.
    """
    logger.info(f"Scanning dataset at {data_dir}...")
    records = []
    
    # Expected structure: Split/Class/Image
    splits = ["Train", "Validation", "Test"]
    classes = ["Fake", "Real"]
    
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
            
        for label in classes:
            class_dir = split_dir / label
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            images = list(class_dir.glob("*.*"))
            logger.info(f"Found {len(images)} images in {split}/{label}")
            
            for img_path in tqdm(images, desc=f"Scanning {split}/{label}", leave=False):
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        mode = img.mode
                        
                    records.append({
                        "path": str(img_path),
                        "filename": img_path.name,
                        "split": split,
                        "label": label,
                        "width": width,
                        "height": height,
                        "aspect_ratio": width / height,
                        "mode": mode,
                        "file_size_kb": img_path.stat().st_size / 1024
                    })
                except Exception as e:
                    logger.error(f"Corrupt image found: {img_path} - {e}")
                    
    df = pd.DataFrame(records)
    logger.info(f"Metadata collection complete. Total records: {len(df)}")
    return df

def analyze_distributions(df: pd.DataFrame):
    """
    Generates plots for class distribution and image properties.
    """
    logger.info("Generating distribution plots...")
    sns.set_theme(style="whitegrid")
    
    # 1. Class Distribution per Split
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="split", hue="label", palette="viridis")
    plt.title("Class Distribution by Split")
    plt.ylabel("Count")
    plt.savefig(OUTPUT_DIR / "class_distribution.png")
    plt.close()
    
    # 2. Image Dimensions Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="width", y="height", hue="split", alpha=0.5)
    plt.title("Image Dimensions (Width vs Height)")
    plt.savefig(OUTPUT_DIR / "image_dimensions.png")
    plt.close()
    
    # 3. Aspect Ratio Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="aspect_ratio", hue="label", kde=True, bins=30)
    plt.title("Aspect Ratio Distribution")
    plt.savefig(OUTPUT_DIR / "aspect_ratio_dist.png")
    plt.close()

def pixel_intensity_analysis(df: pd.DataFrame, sample_size: int = 1000):
    """
    Analyzes pixel intensities on a random sample of images.
    """
    logger.info(f"Running pixel intensity analysis on {sample_size} random images...")
    
    if len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=42)
    else:
        sample_df = df
        
    pixel_stats = {"r_mean": [], "g_mean": [], "b_mean": []}
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Pixel Analysis"):
        try:
            with Image.open(row["path"]) as img:
                img = img.convert("RGB")
                arr = np.array(img)
                
                pixel_stats["r_mean"].append(arr[:, :, 0].mean())
                pixel_stats["g_mean"].append(arr[:, :, 1].mean())
                pixel_stats["b_mean"].append(arr[:, :, 2].mean())
        except:
            pass
            
    # Plot RGB Intensity Distributions
    plt.figure(figsize=(12, 6))
    sns.kdeplot(pixel_stats["r_mean"], color="red", label="Red Channel", fill=True, alpha=0.3)
    sns.kdeplot(pixel_stats["g_mean"], color="green", label="Green Channel", fill=True, alpha=0.3)
    sns.kdeplot(pixel_stats["b_mean"], color="blue", label="Blue Channel", fill=True, alpha=0.3)
    plt.title("Average Pixel Intensity Distribution (RGB)")
    plt.xlabel("Pixel Value (0-255)")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "pixel_intensity_dist.png")
    plt.close()

def generate_summary_report(df: pd.DataFrame):
    """
    Writes a text summary of the EDA findings.
    """
    report_path = OUTPUT_DIR / "eda_summary.txt"
    
    with open(report_path, "w") as f:
        f.write("="*50 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total Images: {len(df)}\n")
        f.write(f"Unique Resolutions: {len(df.groupby(['width', 'height']))}\n")
        f.write(f"Most Common Resolution: {df.groupby(['width', 'height']).size().idxmax()}\n\n")
        
        f.write("-" * 30 + "\n")
        f.write("CLASS BALANCE\n")
        f.write("-" * 30 + "\n")
        balance = df.groupby(["split", "label"]).size().unstack(fill_value=0)
        f.write(balance.to_markdown())
        f.write("\n\n")
        
        f.write("-" * 30 + "\n")
        f.write("IMAGE PROPERTIES\n")
        f.write("-" * 30 + "\n")
        f.write(df[["width", "height", "aspect_ratio", "file_size_kb"]].describe().to_markdown())
        f.write("\n")
        
    logger.info(f"Summary report saved to {report_path}")

def main():
    logger.info("Starting EDA Process...")
    
    # 1. Collect Data
    df = collect_metadata(DATA_DIR)
    if df.empty:
        logger.error("No images found! Check DATA_DIR path.")
        return
        
    # Save raw metadata
    df.to_csv(OUTPUT_DIR / "dataset_metadata.csv", index=False)
    
    # 2. Visualizations
    analyze_distributions(df)
    
    # 3. Pixel Analysis
    pixel_intensity_analysis(df)
    
    # 4. Text Report
    generate_summary_report(df)
    
    logger.info("EDA Complete. Check logs/eda_reports/ for results.")

if __name__ == "__main__":
    main()
