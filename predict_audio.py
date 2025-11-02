import librosa
import numpy as np
from joblib import load

# Load model
model = load("emotion_model.joblib")

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.hstack((mfcc_mean, mfcc_std))

# File to test
file_path = "myvoice.wav"

# Extract features
features = extract_features(file_path).reshape(1, -1)

# Predict
prediction = model.predict(features)[0]

print("\n🎤 Emotion Prediction Result:")
print("➡️ Emotion:", prediction)
