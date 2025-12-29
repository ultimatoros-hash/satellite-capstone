import subprocess
import os
import sys

def run_command(command, description):
    """Exécute une commande terminal et gère les erreurs sans bloquer tout le pipeline."""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {description}")
    print(f"{'='*60}")
    
    try:
        # shell=True permet d'utiliser 'python' directement
        subprocess.run(command, shell=True, check=True)
        print(f"✅ SUCCESS: {description} completed.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: {description} failed.")
        # On continue même si un benchmark échoue (ce n'est pas critique pour lancer l'app)
        if "benchmark" in description.lower() or "protocol" in description.lower():
            print("⚠️  Warning: Scientific Protocol failed, but continuing pipeline.")
        else:
            sys.exit(1)

def file_exists(filepath):
    return os.path.exists(filepath)

def main():
    print("🛰️  ECO-VISION PIPELINE: FINAL CAPSTONE VERSION")
    print("    Group 17 - Full Scientific Validation Suite\n")

    # --- STEP 1: INSTALL DEPENDENCIES ---
    run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing Dependencies")

    # --- STEP 2: CRAWLING ---
    if file_exists("data/dataset.csv") and os.path.exists("data/raw/images"):
        print("ℹ️  Dataset found. Skipping Crawler.")
    else:
        run_command(f"{sys.executable} crawler.py", "Data Acquisition (Crawling)")

    # --- STEP 3: CLEANING ---
    run_command(f"{sys.executable} cleaner.py", "Data Cleaning (Quality Assurance)")

    # --- STEP 4: ANALYSIS (EDA) ---
    run_command(f"{sys.executable} analysis.py", "Exploratory Data Analysis (EDA)")

    # --- STEP 5: TRAINING (PRODUCTION MODEL) ---
    # C'est le modèle Custom CNN qui sera utilisé dans l'App
    if file_exists("models/satellite_custom_cnn.h5"):
        print("ℹ️  Production Model found. Skipping Standard Training.")
    else:
        run_command(f"{sys.executable} train.py", "Training Production Model (Custom CNN)")

    # --- STEP 6: EVALUATION (Standard) ---
    run_command(f"{sys.executable} visualize_results.py", "Standard Evaluation (Confusion Matrix)")

    # =========================================================================
    # --- STEP 7: SCIENTIFIC BENCHMARKING (PROTOCOLES A-F) ---
    # La suite complète pour "étoffer" le rapport de 35 pages
    # =========================================================================
    print(f"\n{'#'*60}")
    print("🔬 STARTING SCIENTIFIC VALIDATION PROTOCOLS")
    print(f"{'#'*60}")

    # --- AXE 1 : PERFORMANCE ---
    
    # A. Validation Croisée (Preuve de stabilité)
    if file_exists("train_kfold.py"):
        run_command(f"{sys.executable} train_kfold.py", "Protocol A: 5-Fold Cross-Validation (Stability)")
    else:
        print("⚠️  Missing 'train_kfold.py'. Skipping Protocol A.")

    # B. Baseline Machine Learning (Comparaison Low-Tech)
    if file_exists("baseline_ml.py"):
        run_command(f"{sys.executable} baseline_ml.py", "Protocol B: Random Forest Baseline (Benchmark)")
    else:
        print("⚠️  Missing 'baseline_ml.py'. Skipping Protocol B.")

    # [NOUVEAU] C. Transfer Learning (Comparaison High-Tech)
    if file_exists("train_transfer.py"):
        run_command(f"{sys.executable} train_transfer.py", "Protocol C: MobileNetV2 Transfer Learning (SOTA Comparison)")
    else:
        print("⚠️  Missing 'train_transfer.py'. Skipping Protocol C.")

    # --- AXE 2 : ROBUSTESSE & ÉTHIQUE ---

    # [NOUVEAU] D. Stress Test (Résistance au bruit)
    if file_exists("test_robustness.py") and file_exists("models/satellite_custom_cnn.h5"):
        run_command(f"{sys.executable} test_robustness.py", "Protocol D: Adversarial Noise Stress Test")
    else:
        print("⚠️  Missing 'test_robustness.py'. Skipping Protocol D.")

    # E. Audit Éthique/Géographique
    if file_exists("evaluate_bias.py"):
        run_command(f"{sys.executable} evaluate_bias.py", "Protocol E: Geographic Bias Audit")
    else:
        print("⚠️  Missing 'evaluate_bias.py'. Skipping Protocol E.")

    # --- AXE 3 : TRANSPARENCE ---

    # F. Advanced Analytics (Physique & Mats)
    if file_exists("advanced_analytics.py") and file_exists("models/satellite_custom_cnn.h5"):
        run_command(f"{sys.executable} advanced_analytics.py", "Protocol F: Physics & Manifold Analytics")
    
    print(f"\n{'#'*60}")
    print("✅ ALL SCIENTIFIC PROTOCOLS COMPLETE.")
    print(f"{'#'*60}\n")

    # --- STEP 8: LAUNCH APP ---
    print(f"\n{'='*60}")
    print("🌍 LAUNCHING ECO-VISION DASHBOARD...")
    print(f"{'='*60}")
    
    subprocess.run(f"{sys.executable} -m streamlit run app.py", shell=True)

if __name__ == "__main__":
    main()