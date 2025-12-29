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

# Create plot directory if not exists
os.makedirs(PLOT_DIR, exist_ok=True)

def load_data_and_model():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Run train.py first.")
        return None, None, None

    print("Loading Model & Validation Data...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load validation data (Shuffle=False is crucial for Evaluation)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    class_names = val_ds.class_names
    return model, val_ds, class_names

def run_evaluation(model, dataset, class_names):
    """
    Runs inference once to generate:
    1. Text Classification Report (Console Output)
    2. Confusion Matrix (Saved Plot)
    """
    print("Running Inference for Evaluation...")
    
    y_true = []
    y_pred = []
    
    # Iterate over the entire dataset
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # --- 1. Text Metrics ---
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    print("\n" + "="*60)
    print(f"🏆 MODEL ACCURACY: {acc*100:.2f}%")
    print("="*60)
    print(report)
    print("="*60)
    
    # Save text report to file
    with open(os.path.join(PLOT_DIR, "metrics_report.txt"), "w") as f:
        f.write(report)

    # --- 2. Confusion Matrix Plot ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    save_path = os.path.join(PLOT_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"✅ Confusion Matrix saved to {save_path}")
    plt.close()

def plot_tsne(model, dataset, class_names):
    print("Generating t-SNE Plot (The 'Brain Map')...")
    
    # Wake up model
    dummy = tf.zeros((1, 128, 128, 3))
    _ = model(dummy)

    # Create feature extractor
    feature_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    features = []
    labels = []
    
    # Limit to ~300 images for speed
    for img_batch, label_batch in dataset.take(10):
        batch_features = feature_model.predict(img_batch, verbose=0)
        features.extend(batch_features)
        labels.extend(label_batch.numpy())
        
    features = np.array(features)
    labels = np.array(labels)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    label_names = [class_names[i] for i in labels]
    
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1], 
        hue=label_names, palette='bright', s=100, alpha=0.8
    )
    plt.title("t-SNE Projection")
    plt.legend(title="True Class")
    
    save_path = os.path.join(PLOT_DIR, 'tsne_clusters.png')
    plt.savefig(save_path)
    print(f"✅ t-SNE saved to {save_path}")
    plt.close()

def plot_sample_grid(model, dataset, class_names):
    print("Generating Prediction Grid...")
    plt.figure(figsize=(12, 12))
    
    # Shuffle temporarily for the grid view
    shuffled_ds = dataset.shuffle(1000)
    
    for images, labels in shuffled_ds.take(1):
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        
        for i in range(16):
            if i >= len(images): break
            ax = plt.subplot(4, 4, i + 1)
            img = images[i].numpy().astype("uint8")
            plt.imshow(img)
            
            true_l = class_names[labels[i]]
            pred_l = class_names[pred_labels[i]]
            color = 'green' if true_l == pred_l else 'red'
            
            plt.title(f"T: {true_l}\nP: {pred_l}", color=color, fontsize=10)
            plt.axis("off")
            
    save_path = os.path.join(PLOT_DIR, 'prediction_grid.png')
    plt.savefig(save_path)
    print(f"✅ Prediction Grid saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    model, val_ds, classes = load_data_and_model()
    if model:
        # 1. Metrics & Confusion Matrix
        run_evaluation(model, val_ds, classes)
        
        # 2. t-SNE
        plot_tsne(model, val_ds, classes)
        
        # 3. Visual Grid
        plot_sample_grid(model, val_ds, classes)