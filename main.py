"""
FRESH vs ROTTEN CLASSIFICATION PROJECT (Updated)
================================================

"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / "My_Project_Data"

if not (DATA_DIR / "Fresh_Images").exists():
    DATA_DIR = SCRIPT_DIR.parent / "My_Project_Data"

FRESH_DIR = DATA_DIR / "Fresh_Images"
ROTTEN_DIR = DATA_DIR / "Rotten_Images"

IMG_SIZE = 100
TEST_SIZE = 0.20
RANDOM_STATE = 42

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print(f"Looking for data in: {DATA_DIR}\n")


def load_data():
    if not FRESH_DIR.exists() or not ROTTEN_DIR.exists():
        print("❌ Data folders not found!")
        print(f"Expected: {DATA_DIR}/Fresh_Images and {DATA_DIR}/Rotten_Images")
        exit()

    images, labels = [], []
    
    print("📂 Loading Fresh images...")
    for filename in os.listdir(FRESH_DIR):
        img = cv2.imread(str(FRESH_DIR / filename))
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img.astype('float32') / 255.0)
            labels.append(0)
    
    print("📂 Loading Rotten images...")
    for filename in os.listdir(ROTTEN_DIR):
        img = cv2.imread(str(ROTTEN_DIR / filename))
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img.astype('float32') / 255.0)
            labels.append(1)
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"✅ Dataset Loaded: {X.shape[0]} images")
    print(f"   Fresh: {np.sum(y==0)} | Rotten: {np.sum(y==1)}\n")
    
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_cnn(X_train, X_test, y_train, y_test):
    print("\n" + "="*70)
    print("TRAINING CNN (Max 12 Epochs - Reduced to Reduce Overfitting)")
    print("="*70)
    
    model = build_cnn_model()
    model.summary()
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=4, 
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=12,                    
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🎯 CNN Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    model.save('models/cnn_model.h5')
    print("💾 CNN model saved to 'models/cnn_model.h5'")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Model Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('CNN Model Loss')
    plt.legend()
    plt.savefig('results/cnn_training_history.png')
    plt.show()
    
    return model


def train_svm(X_train, X_test, y_train, y_test):
    print("\n" + "="*70)
    print("TRAINING SVM")
    print("="*70)
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    svm_model = SVC(kernel='linear', random_state=RANDOM_STATE)
    svm_model.fit(X_train_flat, y_train)
    
    y_pred = svm_model.predict(X_test_flat)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 SVM Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_test, y_pred, target_names=['Fresh', 'Rotten']))
    
    joblib.dump(svm_model, 'models/svm_model.pkl')
    print("💾 SVM model saved!")
    return svm_model


def main():
    print("\n🍎 FRESH vs ROTTEN CLASSIFICATION SYSTEM (Updated)")
    print("="*70)
    
    X_train, X_test, y_train, y_test = load_data()
    
    while True:
        print("\nChoose Model:")
        print("1. SVM")
        print("2. CNN (12 epochs)")
        print("3. Both")
        print("4. Exit")
        
        choice = input("\nEnter choice (1/2/3/4): ").strip()
        
        if choice == '1':
            train_svm(X_train, X_test, y_train, y_test)
        elif choice == '2':
            train_cnn(X_train, X_test, y_train, y_test)
        elif choice == '3':
            train_svm(X_train, X_test, y_train, y_test)
            train_cnn(X_train, X_test, y_train, y_test)
        elif choice == '4':
            print("\n✅ Done! Models saved in 'models/' folder.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()