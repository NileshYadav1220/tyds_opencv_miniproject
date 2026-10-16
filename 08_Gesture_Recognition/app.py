# Easy hand landmark detection with MediaPipe
import cv2
import mediapipe as mp

hands_module = mp.solutions.hands
draw = mp.solutions.drawing_utils
camera = cv2.VideoCapture(0)

with hands_module.Hands(max_num_hands=1) as hands:
    print("Show your hand. Press q to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hand, hands_module.HAND_CONNECTIONS)
            cv2.putText(frame, "Hand detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
