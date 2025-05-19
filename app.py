from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO  # Import YOLO from ultralytics
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the YOLO model
try:
    model = YOLO(r'D:\bagathesh\pythonproject\runs\train\yolo_model_cuda\weights\best.pt')  # Update with the correct model path
    print("YOLO Model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# Define class names for road traffic signs
class_names = ['No_u-turn','No-parking','Uneven_road','No_heavy_vehicle']

def traffic_sign_detection(image_path):
    try:
        print(f"Processing image: {image_path}")

        # Perform YOLO inference on the image
        results = model(image_path)

        # Extract the detected objects
        if results and results[0].boxes:
            boxes = results[0].boxes
            detected_classes = [class_names[int(box.cls)] for box in boxes]
            confidences = [float(box.conf) * 100 for box in boxes]

            # Return the most confident prediction
            max_conf_index = confidences.index(max(confidences))
            predicted_class = detected_classes[max_conf_index]
            confidence = confidences[max_conf_index]
        else:
            predicted_class = "No Traffic Sign Detected"
            confidence = 0.0

        return predicted_class, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction", 0.0


@app.route("/", methods=["GET", "POST"])
def upload_page():
    if request.method == "POST":
        if 'file' not in request.files:
            print("No file part in the request.")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            print("No file selected.")
            return redirect(request.url)

        if file:
            try:
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                predicted_class, confidence = traffic_sign_detection(file_path)
                return redirect(
                    url_for("result_page", prediction=predicted_class, confidence=confidence, image_path=file.filename))
            except Exception as e:
                print(f"Error saving or processing the file: {e}")
                return "Error saving or processing the file."

    return render_template("index.html")


@app.route("/result")
def result_page():
    prediction = request.args.get("prediction")
    confidence = request.args.get("confidence")
    image_path = request.args.get("image_path")
    return render_template("result.html", prediction=prediction, confidence=confidence, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
