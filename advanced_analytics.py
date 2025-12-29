import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy import fftpack
from lime import lime_image
from skimage.segmentation import mark_boundaries

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

def run_fft_analysis(images, class_names, labels):
    """Performs Fourier Transform Analysis"""
    print("Running FFT (Frequency Domain) Analysis...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Get one image per class
    unique_classes = np.unique(labels)
    for i, cls_idx in enumerate(unique_classes):
        # Find first image of this class
        idx = np.where(labels == cls_idx)[0][0]
        img = images[idx].mean(axis=2) # Convert to grayscale
        
        # Perform 2D FFT
        fft2 = fftpack.fft2(img)
        # Shift zero frequency to center
        fft2_shifted = fftpack.fftshift(fft2)
        # Calculate Magnitude Spectrum (Log scale)
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
    
    # Flatten images for Scikit-Learn (N, 128*128*3)
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
    model = tf.keras.models.load_model(model_path)
    
    # Find first Conv2D layer
    for layer in model.layers:
        if 'conv' in layer.name:
            filters, biases = layer.get_weights()
            # Normalize to 0-1 for plotting
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            
            # Plot first 16 filters
            n_filters = 16
            fig, axes = plt.subplots(2, 8, figsize=(20, 5))
            for i in range(n_filters):
                f = filters[:, :, :, i]
                ax = axes[i//8, i%8]
                ax.imshow(f) # Plot RGB filter
                ax.axis('off')
            plt.suptitle(f"Layer {layer.name} Filter Visualization")
            plt.savefig(f"{PLOT_DIR}/cnn_filters.png")
            print("✅ Filter Viz Saved.")
            return

def run_lime_explanation(model_path, images, labels, class_names):
    """Runs LIME (Local Interpretable Model-agnostic Explanations)"""
    print("Running LIME Explanation...")
    model = tf.keras.models.load_model(model_path)
    explainer = lime_image.LimeImageExplainer()
    
    # Pick a random Urban image
    idx = np.where(labels == class_names.index('urban'))[0][0]
    img = images[idx].astype('double')
    
    explanation = explainer.explain_instance(
        img, model.predict, top_labels=1, hide_color=0, num_samples=1000
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title("LIME Explanation (Superpixels)")
    plt.axis('off')
    plt.savefig(f"{PLOT_DIR}/lime_explanation.png")
    print("✅ LIME Plot Saved.")

if __name__ == "__main__":
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    
    imgs, lbls, names = load_sample_data(200)
    
    # 1. Physics Analysis
    run_fft_analysis(imgs, names, lbls)
    
    # 2. Statistical Analysis
    run_manifold_learning(imgs, lbls, names)
    
    # 3. Model Analysis
    if os.path.exists(MODEL_PATH):
        visualize_cnn_filters(MODEL_PATH)
        run_lime_explanation(MODEL_PATH, imgs, lbls, names)
    else:
        print("Skipping Model Analysis (Model not found)")