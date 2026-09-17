import os

os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

import cv2
import numpy as np
import base64
import time

app = Flask(__name__)

# ==========================================
# LOAD YOLO MODEL
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolov8n.pt"
)

print("===================================")
print("Loading YOLO model")
print("Model:", MODEL_PATH)
print("===================================")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")


# ==========================================
# HOME
# ==========================================

@app.route("/")
def index():
    return render_template("index.html")


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "YOLOv8n"
    })


# ==========================================
# OBJECT DETECTION
# ==========================================

@app.route("/detect", methods=["POST"])
def detect():

    start_time = time.time()

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data received"
            }), 400

        if "image" not in data:
            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400


        # ==================================
        # DECODE IMAGE
        # ==================================

        image_data = data["image"]

        # Handle:
        # data:image/jpeg;base64,XXXX
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)

        np_arr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "success": False,
                "error": "Could not decode image"
            }), 400


        # ==================================
        # YOLO DETECTION
        # ==================================

        results = model.predict(
            source=frame,
            conf=0.20,
            imgsz=640,
            verbose=False
        )


        result = results[0]


        # ==================================
        # DRAW DETECTIONS
        # ==================================

        annotated = result.plot()


        # ==================================
        # DETECTION DATA
        # ==================================

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                class_name = model.names[
                    class_id
                ]

                detections.append({

                    "class_id": class_id,

                    "class_name":
                        class_name,

                    "confidence":
                        round(
                            confidence,
                            3
                        )

                })


        # ==================================
        # ENCODE OUTPUT IMAGE
        # ==================================

        success, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                80
            ]
        )

        if not success:

            return jsonify({
                "success": False,
                "error": "Could not encode image"
            }), 500


        output_image = base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")


        processing_time = (
            time.time() - start_time
        )


        print(
            f"Detection completed: "
            f"{len(detections)} objects | "
            f"{processing_time:.2f}s"
        )


        # ==================================
        # RESPONSE
        # ==================================

        return jsonify({

            "success": True,

            "image":
                "data:image/jpeg;base64,"
                + output_image,

            "detections":
                detections,

            "processing_time":
                round(
                    processing_time,
                    2
                )

        })


    except Exception as e:

        print(
            "Detection error:",
            repr(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ==========================================
# LOCAL DEVELOPMENT
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
