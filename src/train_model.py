import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from extract_features import extract_features

def prepare_dataset(data_dir):
    """Load dataset, extract features, and create labels."""
    features = []
    labels = []
    label_map = {"A": 0, "B": 1, "C": 2}

    for speaker, label in label_map.items():
        speaker_dir = os.path.join(data_dir, speaker)
        for file_name in os.listdir(speaker_dir):
            file_path = os.path.join(speaker_dir, file_name)
            mfcc = extract_features(file_path)
            features.append(mfcc)
            labels.append(label)

    return np.array(features), np.array(labels)

def build_model(input_shape):
    """Define the CNN model architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir, model_save_path):
    """Train the model and save it."""
    X, y = prepare_dataset(data_dir)
    X = X[..., np.newaxis]  # Add channel dimension
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(input_shape=X_train.shape[1:])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=16)

    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    data_dir = "../dataset"
    model_save_path = "../model/speaker_diarization_model.h5"
    train_model(data_dir, model_save_path)
