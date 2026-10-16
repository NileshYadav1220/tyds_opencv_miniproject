# Easy multi-object tracking using OpenCV trackers
import cv2

video_path = input("Video path (press Enter for webcam): ").strip()
camera = cv2.VideoCapture(video_path if video_path else 0)

if not camera.isOpened():
    print("Could not open video or webcam.")
    raise SystemExit

ok, frame = camera.read()
if not ok:
    print("Could not read first frame.")
    raise SystemExit

boxes = cv2.selectROIs("Select objects, then press Enter", frame)
trackers = []

for box in boxes:
    x, y, w, h = map(int, box)
    if w > 0 and h > 0:
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(frame, (x, y, w, h))
        trackers.append(tracker)

print("Press q to quit.")
while True:
    ok, frame = camera.read()
    if not ok:
        break

    for number, tracker in enumerate(trackers, start=1):
        found, box = tracker.update(frame)
        if found:
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Object {number}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
