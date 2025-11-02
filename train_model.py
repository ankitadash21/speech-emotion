import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# Load features and labels
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print("✅ Loaded dataset")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create SVM model
model = SVC(kernel="rbf", C=10, gamma="scale")

print("🚀 Training model...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧠 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from joblib import dump
dump(model, "emotion_model.joblib")
print("✅ Model saved as emotion_model.joblib")