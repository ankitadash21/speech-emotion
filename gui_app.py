import tkinter as tk
import sounddevice as sd
import wavio
import librosa
import numpy as np
from joblib import load

model = load("emotion_model.joblib")

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.hstack((mfcc_mean, mfcc_std))

def record_audio():
    status_label.config(text="🎤 Recording 3 seconds...")
    window.update()

    fs = 44100  # Sample rate
    duration = 3  # seconds
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    wavio.write("live_voice.wav", recording, fs, sampwidth=2)
    status_label.config(text="✅ Recording saved!")

def predict_emotion():
    try:
        features = extract_features("live_voice.wav").reshape(1, -1)
        prediction = model.predict(features)[0]
        result_label.config(text=f"Emotion: {prediction}")
    except:
        result_label.config(text="⚠️ Please record first!")

# GUI Window
window = tk.Tk()
window.title("Speech Emotion Detector")

record_btn = tk.Button(window, text="🎤 Record Voice", command=record_audio, width=25)
record_btn.pack(pady=10)

predict_btn = tk.Button(window, text="🧠 Predict Emotion", command=predict_emotion, width=25)
predict_btn.pack(pady=10)

status_label = tk.Label(window, text="")
status_label.pack(pady=5)

result_label = tk.Label(window, text="Emotion: ", font=("Arial", 16))
result_label.pack(pady=10)

window.mainloop()
