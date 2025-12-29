import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import pathlib
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# --- CONFIG ---
DATA_DIR = pathlib.Path("data/raw/images")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
N_FOLDS = 5

def get_model(num_classes):
    """Reconstruit un modèle vierge à chaque fold"""
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        layers.Rescaling(1./255),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_kfold():
    print(f"🚀 Démarrage de la {N_FOLDS}-Fold Cross-Validation...")
    
    # 1. Récupération manuelle des chemins d'images (nécessaire pour KFold)
    image_paths = list(DATA_DIR.glob('*/*.jpg')) # Assure-toi que c'est .jpg ou *
    image_paths = [str(path) for path in image_paths]
    np.random.shuffle(image_paths)
    
    # Extraction des labels depuis les dossiers
    class_names = sorted([item.name for item in DATA_DIR.glob('*') if item.is_dir()])
    class_indices = {name: i for i, name in enumerate(class_names)}
    labels = [class_indices[pathlib.Path(path).parent.name] for path in image_paths]
    
    X = np.array(image_paths)
    y = np.array(labels)
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_no = 1
    accuracies = []

    for train_index, val_index in kfold.split(X):
        print(f"\n🔄 FOLD {fold_no}/{N_FOLDS}")
        
        # Séparation des chemins de fichiers
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Création des tf.data.Dataset à la volée
        def process_path(file_path, label):
            img = tf.io.read_file(file_path)
            img = tf.io.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            return img, label

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
                     .map(process_path).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
                   .map(process_path).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Entraînement
        model = get_model(len(class_names))
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=0)
        
        # Évaluation
        loss, acc = model.evaluate(val_ds, verbose=0)
        print(f"   ✅ Score du Fold {fold_no}: {acc*100:.2f}%")
        accuracies.append(acc)
        fold_no += 1

    # RÉSULTATS SCIENTIFIQUES
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    
    print(f"\n{'='*40}")
    print(f"🏆 RÉSULTAT FINAL (Scientific Rigor)")
    print(f"   Précision Moyenne : {mean_acc:.2f}%")
    print(f"   Écart-type (Stabilité) : ± {std_acc:.2f}%")
    print(f"{'='*40}")

if __name__ == "__main__":
    run_kfold()