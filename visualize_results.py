import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

# --- CONFIG ---
DATA_DIR = "data/raw/images"
MODEL_PATH = "models/satellite_custom_cnn.h5"
PLOT_DIR = "data/plots"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# PALETTE OFFICIELLE (Cohérence graphique)
PALETTE = {
    "snow": "#d1d5db",   # Light Gray
    "water": "#3b82f6",  # Blue
    "forest": "#16a34a", # Green
    "urban": "#ef4444",  # Red
    "desert": "#f97316"  # Orange
}

def load_data_and_model():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Run train.py first.")
        return None, None, None

    print("Loading Model & Validation Data...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    class_names = val_ds.class_names
    return model, val_ds, class_names

def run_evaluation(model, dataset, class_names):
    print("Running Inference for Evaluation...")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    y_true = []
    y_pred = []
    
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    print(f"🏆 MODEL ACCURACY: {acc*100:.2f}%")
    
    # --- FIX "N/A" : ÉCRITURE LISIBLE PAR MACHINE ---
    with open(os.path.join(PLOT_DIR, "metrics_report.txt"), "w") as f:
        f.write(f"GLOBAL_ACCURACY: {acc:.5f}\n") # Ligne clé pour l'app
        f.write("\n" + report)

    # --- CONFUSION MATRIX ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix.png'))
    plt.close()

def plot_tsne(model, dataset, class_names):
    print("Generating t-SNE Plot...")
    
    # Feature Extractor
    feature_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    features = []
    labels = []
    
    # On prend un sous-ensemble pour la vitesse (ex: 10 batchs)
    for img_batch, label_batch in dataset.take(15):
        batch_features = feature_model.predict(img_batch, verbose=0)
        features.extend(batch_features)
        labels.extend(label_batch.numpy())
        
    features = np.array(features)
    labels = np.array(labels)
    label_names = [class_names[i] for i in labels]
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    # --- FIX COULEURS : Utilisation de la PALETTE officielle ---
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1], 
        hue=label_names, palette=PALETTE, s=80, alpha=0.8, edgecolor="k"
    )
    plt.title("t-SNE Projection (Feature Space)")
    plt.legend(title="Class")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, 'tsne_clusters.png'))
    plt.close()

if __name__ == "__main__":
    model, val_ds, classes = load_data_and_model()
    if model:
        run_evaluation(model, val_ds, classes)
        plot_tsne(model, val_ds, classes)