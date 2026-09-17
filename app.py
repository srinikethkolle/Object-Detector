from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load YOLO model once when the server starts
model = YOLO("yolov8n.pt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400

        # Remove the "data:image/jpeg;base64," part
        image_data = data["image"].split(",")[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)

        # Convert bytes to NumPy array
        np_arr = np.frombuffer(image_bytes, np.uint8)

        # Convert to OpenCV image
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "success": False,
                "error": "Could not decode image"
            }), 400

        # Run YOLO detection
        results = model(frame)

        # Draw detections
        annotated = results[0].plot()

        # Encode annotated frame as JPEG
        success, buffer = cv2.imencode(".jpg", annotated)

        if not success:
            return jsonify({
                "success": False,
                "error": "Could not encode image"
            }), 500

        # Convert to base64
        output_image = base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")

        # Get detection information
        detections = []

        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                detections.append({
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": round(confidence, 2)
                })

        return jsonify({
            "success": True,
            "image": "data:image/jpeg;base64," + output_image,
            "detections": detections
        })

    except Exception as e:
        print("Error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )