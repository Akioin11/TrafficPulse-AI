import cv2
import torch

# Load YOLOv5 model (GPU)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to('cuda')

# Webcam capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For Windows webcam handling

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Render results on frame
    results.render()
    output = results.ims[0]
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Show output
    cv2.imshow('YOLOv5 🚦', output)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
