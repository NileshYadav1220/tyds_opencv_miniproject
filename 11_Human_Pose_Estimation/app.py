# Easy human pose estimation with MediaPipe
import cv2
import mediapipe as mp

pose_module = mp.solutions.pose
draw = mp.solutions.drawing_utils
camera = cv2.VideoCapture(0)

with pose_module.Pose() as pose:
    print("Stand in front of camera. Press q to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            draw.draw_landmarks(frame, result.pose_landmarks,
                                pose_module.POSE_CONNECTIONS)
            cv2.putText(frame, "Person detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Human Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
