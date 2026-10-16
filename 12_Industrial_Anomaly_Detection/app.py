# Easy unsupervised anomaly detection using Isolation Forest
import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

def get_features(folder):
    features = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        image = cv2.imread(path)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (32, 32))
            features.append(gray.flatten() / 255.0)
    return np.array(features)

normal_folder = input("Folder with normal/good images: ")
X = get_features(normal_folder)

if len(X) < 5:
    print("Please add at least 5 normal images.")
    raise SystemExit

model = IsolationForest(contamination="auto", random_state=42)
model.fit(X)
joblib.dump(model, "anomaly_model.joblib")
print("Model trained and saved.")

test_path = input("Image to check: ")
image = cv2.imread(test_path)
if image is None:
    print("Image not found.")
    raise SystemExit

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (32, 32))
prediction = model.predict([gray.flatten() / 255.0])[0]

if prediction == 1:
    print("Result: Normal")
else:
    print("Result: Possible anomaly")
