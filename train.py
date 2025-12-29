import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# --- CONFIG ---
DATA_DIR = "data/raw/images"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25 # Un peu plus long pour bien apprendre
PLOT_DIR = "data/plots"

# --- DATA AUGMENTATION (VERSION CORRIGÉE & DOUCE) ---
class GentleCloudAugmentation(layers.Layer):
    """Simule des nuages légers pour ne pas détruire l'image."""
    def call(self, inputs, training=True):
        if not training: return inputs
        
        # Bruit plus subtil
        noise = tf.random.uniform(shape=tf.shape(inputs), minval=0.0, maxval=1.0)
        # Nuages moins fréquents (seuil 0.7) et moins opaques (0.2 à 0.4)
        clouds = tf.cast(noise > 0.7, tf.float32) * tf.random.uniform([], 0.2, 0.4)
        
        # On ajoute le blanc (nuage) à l'image
        return tf.clip_by_value(inputs + clouds, 0.0, 1.0)

def plot_training_history(history):
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(PLOT_DIR, "training_curves.png"))

def train():
    print("Loading Dataset...")
    # Chargement
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    
    class_names = train_ds.class_names
    print(f"Target Classes: {class_names}")

    # Poids des classes (Crucial si la neige est rare)
    y_train = []
    for images, labels in train_ds.unbatch():
        y_train.append(labels.numpy())
    
    weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=np.array(y_train)
    )
    class_weights = dict(enumerate(weights))
    print(f"Class Weights: {class_weights}")

    # Performance I/O
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # --- ARCHITECTURE RENFORCÉE (V2) ---
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        layers.Rescaling(1./255),
        
        # Augmentation
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        GentleCloudAugmentation(), # Version douce
        
        # Bloc 1 (Détails fins)
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Bloc 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Bloc 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Bloc 4 (Deep Features - Nouveau)
        layers.Conv2D(256, (3, 3), padding='same', activation='relu', name="last_conv_layer"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Classification
        layers.GlobalAveragePooling2D(), # Plus robuste que Flatten
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4), # Régularisation forte
        layers.Dense(len(class_names), activation='softmax')
    ])

    # Learning Rate décroissant pour affiner la fin
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.9)
    
    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    callbacks_list = [
        callbacks.ModelCheckpoint('models/satellite_custom_cnn.h5', save_best_only=True, monitor='val_accuracy'),
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]
    
    print("🚀 Starting Training (Enhanced Architecture)...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, 
                        class_weight=class_weights, callbacks=callbacks_list)
    
    # Sauvegarde
    with open('models/classes.txt', 'w') as f:
        f.write('\n'.join(class_names))
        
    plot_training_history(history)
    print("✅ Training Complete.")

if __name__ == "__main__":
    train()