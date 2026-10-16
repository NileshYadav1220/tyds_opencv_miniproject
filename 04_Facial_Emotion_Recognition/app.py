# Easy facial expression recognition using DeepFace
import cv2
from deepface import DeepFace

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not available.")
    raise SystemExit

print("Press q to quit.")
while True:
    ok, frame = camera.read()
    if not ok:
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("e"):
        try:
            result = DeepFace.analyze(
                frame, actions=["emotion"],
                enforce_detection=False, silent=True
            )
            if isinstance(result, list):
                result = result[0]
            print("Estimated expression:", result["dominant_emotion"])
        except Exception:
            print("Could not estimate expression.")

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
