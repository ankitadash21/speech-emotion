import librosa
import matplotlib.pyplot as plt

file_path = "sample.wav"

audio, sr = librosa.load(file_path, sr=None)

plt.figure(figsize=(10, 4))
plt.plot(audio)
plt.title(f"Waveform (Sample Rate: {sr} Hz)")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.show()
