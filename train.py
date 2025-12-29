import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from sklearn.utils import class_weight

DATA_DIR = "data/raw/images"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

def train():
    print("Loading Dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    
    class_names = train_ds.class_names
    print(f"Detected Classes: {class_names}")

    # --- FIX BIAS: CALCULATE CLASS WEIGHTS ---
    y_train = []
    for images, labels in train_ds.unbatch():
        y_train.append(labels.numpy())
    y_train = np.array(y_train)
    
    weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(weights))
    print(f"Applied Class Weights: {class_weights}")

    # --- CUSTOM CNN ---
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="last_conv_layer"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weights)
    
    if not os.path.exists('models'): os.makedirs('models')
    model.save('models/satellite_custom_cnn.h5')
    with open('models/classes.txt', 'w') as f:
        f.write('\n'.join(class_names))
        
    print("Training Complete. Model Saved.")

if __name__ == "__main__":
    train()