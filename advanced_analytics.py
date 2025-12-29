import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy import fftpack
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.measure import shannon_entropy

# --- CONFIG ---
DATA_DIR = "data/raw/images"
MODEL_PATH = "models/satellite_custom_cnn.h5"
PLOT_DIR = "data/plots"
IMG_SIZE = (128, 128)

def load_sample_data(num_samples=100):
    """Loads a random subset of images for heavy analysis"""
    print("Loading sample data for advanced analysis...")
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, image_size=IMG_SIZE, batch_size=32, shuffle=True, seed=42
    )
    images, labels = [], []
    class_names = ds.class_names
    for batch_img, batch_lbl in ds.take(num_samples // 32 + 1):
        images.extend(batch_img.numpy())
        labels.extend(batch_lbl.numpy())
    return np.array(images[:num_samples]), np.array(labels[:num_samples]), class_names

def calculate_vari(image):
    """
    Calcule le Visible Atmospherically Resistant Index (VARI).
    Formule : (Green - Red) / (Green + Red - Blue)
    """
    # Normalisation 0-1
    img_norm = image.astype('float32') / 255.0
    R = img_norm[:,:,0]
    G = img_norm[:,:,1]
    B = img_norm[:,:,2]
    
    epsilon = 1e-6
    vari = (G - R) / (G + R - B + epsilon)
    return vari

def run_physical_metrics_analysis(images, labels, class_names):
    """Compare l'Entropie Spatiale et le VARI par classe"""
    print("Running Biophysical Metrics Analysis (Entropy & VARI)...")
    
    metrics = {'entropy': [], 'vari_mean': [], 'label': []}
    
    for img, label in zip(images, labels):
        # 1. Spatial Entropy (Texture Complexity)
        # Convert to grayscale for entropy calculation
        gray = tf.image.rgb_to_grayscale(img).numpy()
        E = shannon_entropy(gray)
        
        # 2. VARI (Vegetation Health)
        V = np.mean(calculate_vari(img))
        
        metrics['entropy'].append(E)
        metrics['vari_mean'].append(V)
        metrics['label'].append(class_names[label])
        
    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Boxplot Entropie
    sns.boxplot(x='label', y='entropy', data=metrics, ax=ax1, palette='magma')
    ax1.set_title("Spatial Entropy (Texture Complexity)\n(Urban should be high, Desert/Water low)")
    ax1.grid(True, alpha=0.3)
    
    # Boxplot VARI
    sns.boxplot(x='label', y='vari_mean', data=metrics, ax=ax2, palette='Greens')
    ax2.set_title("VARI Index (Vegetation Density)\n(Forest should be highest)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/physical_metrics.png")
    print("✅ Biophysical Plots Saved.")

def run_fft_analysis(images, class_names, labels):
    """Performs Fourier Transform Analysis"""
    print("Running FFT (Frequency Domain) Analysis...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    unique_classes = np.unique(labels)
    # Limiter aux 4 premières classes s'il y en a plus
    unique_classes = unique_classes[:4] 

    for i, cls_idx in enumerate(unique_classes):
        # Find first image of this class
        idx = np.where(labels == cls_idx)[0][0]
        img = images[idx].mean(axis=2) # Convert to grayscale
        
        # Perform 2D FFT
        fft2 = fftpack.fft2(img)
        fft2_shifted = fftpack.fftshift(fft2)
        magnitude_spectrum = 20 * np.log(np.abs(fft2_shifted))
        
        ax = axes[i]
        ax.imshow(magnitude_spectrum, cmap='inferno')
        ax.set_title(f"FFT: {class_names[cls_idx]}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/fft_spectrum_analysis.png")
    print("✅ FFT Plot Saved.")

def run_manifold_learning(images, labels, class_names):
    """Runs PCA and UMAP comparison"""
    print("Running Manifold Learning (PCA vs UMAP)...")
    
    # Flatten images for Scikit-Learn
    flat_images = images.reshape(images.shape[0], -1) / 255.0
    
    # 1. PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(flat_images)
    
    # 2. UMAP
    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(flat_images)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    scatter1 = ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)
    ax1.set_title(f"PCA Projection (Variance: {np.sum(pca.explained_variance_ratio_):.2f})")
    ax1.legend(handles=scatter1.legend_elements()[0], labels=class_names)
    
    scatter2 = ax2.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)
    ax2.set_title("UMAP Projection (Non-Linear)")
    
    plt.savefig(f"{PLOT_DIR}/pca_vs_umap.png")
    print("✅ Manifold Learning Plot Saved.")

def visualize_cnn_filters(model_path):
    """Visualizes the first layer kernels"""
    print("Visualizing CNN Filters...")
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        print("Model not found for filter viz.")
        return

    # Find first Conv2D layer
    for layer in model.layers:
        if 'conv' in layer.name:
            filters, biases = layer.get_weights()
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            
            n_filters = min(16, filters.shape[3])
            fig, axes = plt.subplots(2, 8, figsize=(20, 5))
            for i in range(n_filters):
                f = filters[:, :, :, i]
                ax = axes[i//8, i%8]
                ax.imshow(f)
                ax.axis('off')
            plt.suptitle(f"Layer {layer.name} Filter Visualization")
            plt.savefig(f"{PLOT_DIR}/cnn_filters.png")
            print("✅ Filter Viz Saved.")
            return

def run_lime_explanation(model_path, images, labels, class_names):
    """Runs LIME explanation"""
    print("Running LIME Explanation...")
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        return

    explainer = lime_image.LimeImageExplainer()
    
    # Try to find an 'urban' image, else take the first one
    try:
        target_idx = class_names.index('urban')
        idx = np.where(labels == target_idx)[0][0]
    except:
        idx = 0

    img = images[idx].astype('double')
    
    explanation = explainer.explain_instance(
        img, model.predict, top_labels=1, hide_color=0, num_samples=1000
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title(f"LIME Explanation for {class_names[labels[idx]]}")
    plt.axis('off')
    plt.savefig(f"{PLOT_DIR}/lime_explanation.png")
    print("✅ LIME Plot Saved.")

if __name__ == "__main__":
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    
    # Load 200 samples
    imgs, lbls, names = load_sample_data(200)
    
    # 1. Physics Analysis (NEW)
    run_physical_metrics_analysis(imgs, lbls, names)
    run_fft_analysis(imgs, names, lbls)
    
    # 2. Statistical Analysis
    run_manifold_learning(imgs, lbls, names)
    
    # 3. Model Analysis
    if os.path.exists(MODEL_PATH):
        visualize_cnn_filters(MODEL_PATH)
        run_lime_explanation(MODEL_PATH, imgs, lbls, names)
