import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

MODEL_PATH = "models/satellite_custom_cnn.h5"
DATA_DIR = "data/raw/images"
IMG_SIZE = (128, 128)

def add_noise(image, noise_factor):
    """Ajoute du 'grain' aléatoire sur l'image"""
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_factor, dtype=tf.float32)
    noisy_image = image + noise * 255.0 # noise factor est relatif à l'échelle 0-1 ou 0-255
    return tf.clip_by_value(noisy_image, 0.0, 255.0)

def run_stress_test():
    if not os.path.exists(MODEL_PATH): return
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # On prend juste un batch de validation pour le test
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=32
    )
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5] # 0% à 50% de bruit
    accuracies = []
    
    print("🌪️ Démarrage du Stress Test...")
    
    for noise in noise_levels:
        correct = 0
        total = 0
        for images, labels in val_ds:
            # Injection de bruit
            noisy_imgs = add_noise(images, noise)
            preds = model.predict(noisy_imgs, verbose=0)
            pred_labels = np.argmax(preds, axis=1)
            
            correct += np.sum(pred_labels == labels.numpy())
            total += len(labels)
            
        acc = correct / total
        accuracies.append(acc)
        print(f"   Niveau de bruit {noise}: Accuracy = {acc*100:.1f}%")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, accuracies, marker='o', linestyle='-', color='red')
    plt.title("Robustesse du Modèle face au Bruit Capteur")
    plt.xlabel("Intensité du Bruit (Sigma)")
    plt.ylabel("Précision du Modèle")
    plt.grid(True)
    plt.savefig("data/plots/robustness_curve.png")
    print("✅ Courbe de robustesse sauvegardée.")

if __name__ == "__main__":
    run_stress_test()