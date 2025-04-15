import torch
import cv2
import time

# Load YOLOv5s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # Confidence threshold
allowed_classes = ['car', 'truck', 'bus']

cap = cv2.VideoCapture(0)

# Timer and count init
last_check = time.time()
left_count = 0
right_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection every 5 seconds
    now = time.time()
    if now - last_check >= 5:
        results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = results.pandas().xyxy[0]

        # Reset counts
        left_count = 0
        right_count = 0
        frame_center = frame.shape[1] // 2

        for _, row in detections.iterrows():
            if row['name'] in allowed_classes:
                x_center = (row['xmin'] + row['xmax']) / 2
                if x_center < frame_center:
                    left_count += 1
                else:
                    right_count += 1

        print(f"[{time.strftime('%H:%M:%S')}] Left: {left_count} | Right: {right_count}")
        last_check = now

    # Draw HUD (always on)
    frame_center = frame.shape[1] // 2
    cv2.putText(frame, f"Left: {left_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Right: {right_count}", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.line(frame, (frame_center, 0), (frame_center, frame.shape[0]), (0, 0, 255), 2)

    # Show video feed
    cv2.imshow('Smart Lane YOLO View', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
