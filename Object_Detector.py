import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolov8n.pt")

# Open the default webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read(0)

    if not ret:
        print("Failed to read from camera.")
        break

    # Run object detection
    results = model(frame)

    # Draw detection results on the frame
    annotated = results[0].plot()

    # Display the result
    cv2.imshow("Object Detection", annotated)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()