import subprocess
import os
import sys
import time

def run_command(command, description):
    """Runs a terminal command and checks for errors."""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {description}")
    print(f"{'='*60}")
    
    # Run the command and wait for it to finish
    try:
        # shell=True allows using 'pip' and 'python' directly
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ SUCCESS: {description} completed.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: {description} failed.")
        print(f"Error Details: {e}")
        sys.exit(1)

def file_exists(filepath):
    return os.path.exists(filepath)

def main():
    print("🛰️  SATELLITE AI PIPELINE: AUTO-LAUNCH SYSTEM")
    print("    Group 17 - Capstone Project\n")

    # --- STEP 1: INSTALL DEPENDENCIES ---
    # Only run if the user explicitly asks, or just ensure they are present
    print("Checking libraries...")
    run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing Dependencies")

    # --- STEP 2: CRAWLING (Conditional) ---
    # Skip if we already have a massive dataset
    if file_exists("data/dataset.csv") and os.path.exists("data/raw/images"):
        print("ℹ️  Dataset found. Skipping Crawler to save time.")
        print("   (Delete 'data/dataset.csv' if you want to re-crawl)")
    else:
        run_command(f"{sys.executable} crawler.py", "Data Acquisition (Crawling)")

    # --- STEP 3: CLEANING ---
    run_command(f"{sys.executable} cleaner.py", "Data Cleaning (Quality Assurance)")

    # --- STEP 4: ANALYSIS (EDA) ---
    run_command(f"{sys.executable} analysis.py", "Exploratory Data Analysis (Generating Plots)")

    # --- STEP 5: TRAINING (Conditional) ---
    # Skip if model already exists to save 20 mins
    if file_exists("models/satellite_custom_cnn.h5"):
        print("ℹ️  Trained Model found. Skipping Training.")
        print("   (Delete 'models/satellite_custom_cnn.h5' if you want to re-train)")
    else:
        run_command(f"{sys.executable} train.py", "Model Training (CNN)")

# --- STEP 6: EVALUATION & VISUALIZATION ---
    # Calculates Metrics, Confusion Matrix, t-SNE, and Grid all in one go
    run_command(f"{sys.executable} visualize_results.py", "Model Evaluation & Visualization")

    # --- STEP 7: LAUNCH APP ---
    print(f"\n{'='*60}")
    print("🌍 LAUNCHING GEOSENTINEL DASHBOARD...")
    print(f"{'='*60}")
    
    # Launch Streamlit
    subprocess.run(f"{sys.executable} -m streamlit run app.py", shell=True)

if __name__ == "__main__":
    main()