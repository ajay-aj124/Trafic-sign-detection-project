from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import torch

app = Flask(__name__)

# Load the YOLO model
try:
    model = YOLO(r'D:\bagathesh\pythonproject\runs\train\yolo_model_cuda\weights\best.pt')
    print("YOLO Model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# Define class names for road traffic signs
class_names = ['No_u-turn','No-parking','Uneven_road','No_heavy_vehicle']

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO inference
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls_id = int(box.cls)
                label = f"{class_names[cls_id]}: {conf:.2f}"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
