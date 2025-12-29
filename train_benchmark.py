import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os
import numpy as np

# --- CONFIG ---
DATA_DIR = "data/raw/images"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_DIR = "models"

def train_benchmark():
    print(f"\n{'='*40}")
    print("🚀 TRAINING BENCHMARK MODEL: MOBILENET V2")
    print(f"{'='*40}")

    if not os.path.exists(DATA_DIR):
        print("❌ Data directory not found.")
        return

    # Load Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    class_names = train_ds.class_names
    
    # --- TRANSFER LEARNING ---
    # Load MobileNetV2 without the top layer (headless)
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,), 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base layers
    
    # Add new head
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Starting Training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    save_path = os.path.join(MODEL_DIR, 'mobilenet_benchmark.h5')
    model.save(save_path)
    
    final_acc = history.history['val_accuracy'][-1] * 100
    print(f"✅ Benchmark Model Saved to {save_path}")
    print(f"🏆 Final Validation Accuracy: {final_acc:.2f}%")

if __name__ == "__main__":
    train_benchmark()