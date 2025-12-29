import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

DATA_DIR = "data/raw/images"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

def run_transfer_learning():
    print("🚀 Training MobileNetV2 (Transfer Learning)...")
    
    # 1. Chargement des données
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    
    class_names = train_ds.class_names
    
    # 2. Le Modèle Pré-entraîné (MobileNetV2)
    # include_top=False : On enlève la dernière couche (les 1000 classes d'ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet'
    )
    base_model.trainable = False # On gèle le "cerveau" pour ne pas casser ce qu'il sait déjà

    # 3. On greffe notre tête de lecture
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        layers.Rescaling(1./127.5, offset=-1), # MobileNet attend des pixels entre -1 et 1
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # 4. Entraînement
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    
    # 5. Sauvegarde
    acc = history.history['val_accuracy'][-1] * 100
    print(f"🏁 MobileNetV2 Accuracy: {acc:.2f}%")
    
    # Comparaison immédiate (à mettre dans le rapport)
    print("Comparez ce chiffre avec votre CNN Custom. Si MobileNet est seulement 1% meilleur mais 10x plus gros, votre CNN gagne.")

if __name__ == "__main__":
    run_transfer_learning()