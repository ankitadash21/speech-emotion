import os
import librosa
import numpy as np

# Path to TESS dataset
DATA_DIR = "data/TESS"

# Map keywords to emotion labels
emotion_map = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "pleasant": "pleasant_surprise"
}

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.hstack((mfcc_mean, mfcc_std))

X = []
y = []

# Walk through dataset folders
for root, dirs, files in os.walk(DATA_DIR):
    folder = os.path.basename(root).lower()  # current folder name

    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            file_lower = file.lower()

            label = None

            # ✅ Check folder name first
            for key in emotion_map:
                if key in folder:
                    label = emotion_map[key]
                    break

            # ✅ If not found, check file name
            if label is None:
                for key in emotion_map:
                    if key in file_lower:
                        label = emotion_map[key]
                        break

            # Still no match? skip file
            if label is None:
                print(f"⚠️ Skipping (no emotion found): {file}")
                continue

            # Extract MFCC features
            features = extract_features(file_path)

            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("\n✅ Dataset built successfully!")
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)
print("Example labels:", y[:10])

# Save features + labels
np.save("X_features.npy", X)
np.save("y_labels.npy", y)

print("\n✅ Saved X_features.npy and y_labels.npy")
