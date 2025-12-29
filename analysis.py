import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import numpy as np

# --- CONFIGURATION ---
CSV_PATH = "data/dataset.csv"
IMG_DIR = "data/raw/images"
PLOT_DIR = "data/plots"

def run_analysis():
    # 1. Setup
    if not os.path.exists(CSV_PATH):
        print("❌ Dataset CSV not found. Run crawler.py first.")
        return
    
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded dataset: {len(df)} records")

    # --- PLOT 1: CLASS DISTRIBUTION (Bar Chart) ---
    print("Generating Class Balance Plot...")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title("Class Distribution of Training Data")
    plt.xlabel("Terrain Class")
    plt.ylabel("Number of Images")
    plt.savefig(os.path.join(PLOT_DIR, "class_balance.png"))
    plt.close()

    # --- PLOT 2: GEOGRAPHIC MAP (Scatter Plot) ---
    print("Generating Geographic Map...")
    plt.figure(figsize=(12, 6))
    # Load world map background (optional) or just plot coordinates
    sns.scatterplot(x='longitude', y='latitude', hue='label', data=df, s=20, palette='deep')
    plt.title("Geographic Source Locations of Satellite Tiles")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "geo_map.png"))
    plt.close()

    # --- PLOT 3: SPECTRAL ANALYSIS (Pixel Intensity) ---
    # This explains the "Urban vs Desert" confusion in your report
    print("Generating Spectral Density Plot (This may take a moment)...")
    
    plt.figure(figsize=(10, 6))
    classes = df['label'].unique()
    
    for label in classes:
        # Get up to 50 random images per class to analyze
        class_files = df[df['label'] == label]['filename'].sample(n=min(50, len(df[df['label']==label])), random_state=42)
        intensities = []
        
        for fname in class_files:
            img_path = os.path.join(IMG_DIR, label, fname)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert to grayscale to get "brightness"
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    intensities.extend(gray.flatten())
        
        # Plot Density (KDE)
        if intensities:
            sns.kdeplot(intensities, label=label, fill=True, alpha=0.3)
            
    plt.title("Spectral Density: Pixel Intensity Distribution by Class")
    plt.xlabel("Pixel Brightness (0=Black, 255=White)")
    plt.ylabel("Density")
    plt.legend()
    plt.xlim(0, 255)
    plt.savefig(os.path.join(PLOT_DIR, "spectral_analysis.png"))
    plt.close()

    print(f"✅ All plots saved to '{PLOT_DIR}/'")

if __name__ == "__main__":
    run_analysis()