import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

DATA_DIR = "data/raw/images"
IMG_SIZE = (64, 64) # Plus petit pour le ML classique (sinon trop lent)

def extract_features(image_path):
    """Extrait l'histogramme de couleur (feature simple)"""
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.resize(img, IMG_SIZE)
    
    # Feature 1: Histogramme de couleurs aplati
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def run_baseline():
    print("🤖 Entraînement Baseline (Random Forest)...")
    data, labels = [], []
    classes = os.listdir(DATA_DIR)
    
    for label in classes:
        path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(path): continue
        print(f"   Processing {label}...")
        for f in os.listdir(path)[:500]: # Limite à 500 images pour aller vite
            feat = extract_features(os.path.join(path, f))
            if feat is not None:
                data.append(feat)
                labels.append(label)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # Le modèle "Bête mais méchant"
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n📊 Baseline Accuracy: {acc*100:.2f}%")
    print("Comparez ce chiffre à votre CNN. Le Deep Learning en vaut-il la peine ?")

if __name__ == "__main__":
    run_baseline()