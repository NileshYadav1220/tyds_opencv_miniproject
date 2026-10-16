# Easy apparent-age estimation using DeepFace
import cv2
from deepface import DeepFace

path = input("Enter face image path: ")
image = cv2.imread(path)

if image is None:
    print("Image not found.")
    raise SystemExit

try:
    result = DeepFace.analyze(
        image, actions=["age"], enforce_detection=False, silent=True
    )
    if isinstance(result, list):
        result = result[0]
    print("Estimated apparent age:", result.get("age", "unknown"))
except Exception as error:
    print("Could not estimate age:", error)
