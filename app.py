# from flask import Flask, request, jsonify, render_template_string
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array

# # Load the trained model
# model = load_model("road_condition_detector.h5")

# # Class labels
# class_labels = ["Decks_Cracked", "Decks_NonCracked", "Pavements_Cracked", "Pavements_NonCracked", "Walls_Cracked", "Walls_NonCracked"]

# # Initialize Flask app
# app = Flask(__name__)

# # Dictionary to store prediction counts for summary
# prediction_summary = {label: 0 for label in class_labels}

# # HTML template with buttons
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Road Condition Detector</title>
# </head>
# <body>
#     <h1>Real-Time Road Condition Detector</h1>
#     <p>Use the buttons below to start detection or shut down the app.</p>
#     <button onclick="startDetection()">Start Detection</button>
#     <button onclick="shutdownApp()">Shutdown</button>
#     <script>
#         function startDetection() {
#             fetch('/start_detection', {
#                 method: 'POST',
#                 headers: {'Content-Type': 'application/json'},
#                 body: JSON.stringify({video_source: 0})
#             }).then(response => response.json())
#               .then(data => alert(data.message))
#               .catch(err => alert('Error: ' + err));
#         }

#         function shutdownApp() {
#             fetch('/shutdown', {
#                 method: 'POST'
#             }).then(response => response.text())
#               .then(data => alert(data))
#               .catch(err => alert('Error: ' + err));
#         }
#     </script>
# </body>
# </html>
# """

# @app.route('/')
# def home():
#     return render_template_string(HTML_TEMPLATE)

# @app.route('/start_detection', methods=['POST'])
# def start_detection():
#     # Access camera or video feed
#     video_source = request.json.get('video_source', 0)  # Default to webcam
#     cap = cv2.VideoCapture(video_source)

#     if not cap.isOpened():
#         return jsonify({"message": "Could not open video source"}), 400

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Preprocess the frame for model prediction
#             frame_resized = cv2.resize(frame, (128, 128))  # Resize to model input size
#             frame_array = img_to_array(frame_resized) / 255.0
#             frame_array = np.expand_dims(frame_array, axis=0)

#             # Predict using the model
#             prediction = model.predict(frame_array)
#             predicted_class = class_labels[np.argmax(prediction)]

#             # Update prediction summary
#             prediction_summary[predicted_class] += 1

#             # Display the frame with prediction (optional)
#             cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.imshow("Road Condition Detector", frame)

#             # Exit on 'q' key press
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         # Release the video source and close OpenCV windows
#         cap.release()
#         cv2.destroyAllWindows()

#     return jsonify({"message": "Detection ended. Close the app or check the console for the summary."})

# @app.route('/shutdown', methods=['POST'])
# def shutdown():
#     # Generate a summary of predictions
#     total_predictions = sum(prediction_summary.values())
#     summary = "Summary of Predictions:\n"
#     for label, count in prediction_summary.items():
#         percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
#         summary += f"{label}: {count} frames ({percentage:.2f}%)\n"

#     print("\n" + summary)  # Print summary to console

#     # Shut down the Flask server
#     func = request.environ.get('werkzeug.server.shutdown')
#     if func is None:
#         raise RuntimeError('Not running with the Werkzeug Server')
#     func()

#     return "Server shutting down...\n" + summary

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify, render_template_string, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import threading
import time
import os

# Load the trained model
model = load_model("road_condition_detector.h5")

# Class labels
class_labels = ["Decks_Cracked", "Decks_NonCracked", "Pavements_Cracked", "Pavements_NonCracked", "Walls_Cracked", "Walls_NonCracked"]

# Initialize Flask app
app = Flask(__name__)

# Dictionary to store prediction counts for summary
prediction_summary = {label: 0 for label in class_labels}

# Initialize OpenCV Video Capture
cap = None
is_detecting = False

# HTML template with buttons and video stream
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Road Condition Detector</title>
</head>
<body>
    <h1>Real-Time Road Condition Detector</h1>
    <p>Use the buttons below to start detection or shut down the app.</p>
    <button onclick="startDetection()">Start Detection</button>
    <button onclick="shutdownApp()">Shutdown</button>
    <br><br>
    <img id="videoFeed" width="640" height="480" />
    <script>
        function startDetection() {
            fetch('/start_detection', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({video_source: 0})
            }).then(response => response.json())
              .then(data => alert(data.message))
              .catch(err => alert('Error: ' + err));
        }

        function shutdownApp() {
            fetch('/shutdown', {
                method: 'POST'
            }).then(response => response.text())
              .then(data => alert(data))
              .catch(err => alert('Error: ' + err));
        }

        // Function to update video feed on the webpage
        function updateFeed() {
            var videoFeed = document.getElementById("videoFeed");
            videoFeed.src = "/video_feed?" + new Date().getTime();
            setTimeout(updateFeed, 100); // Update the feed every 100 ms
        }

        updateFeed(); // Start updating the feed
    </script>
</body>
</html>
"""

# Route to serve the video feed to the webpage
@app.route('/video_feed')
def video_feed():
    """Generate and stream video frames to the client."""
    def generate():
        while is_detecting:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame for model prediction
            frame_resized = cv2.resize(frame, (128, 128))  # Resize to model input size
            frame_array = img_to_array(frame_resized) / 255.0
            frame_array = np.expand_dims(frame_array, axis=0)

            # Predict using the model
            prediction = model.predict(frame_array)
            predicted_class = class_labels[np.argmax(prediction)]

            # Update prediction summary
            prediction_summary[predicted_class] += 1

            # Draw prediction on the frame
            cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert frame to JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            # Yield the frame in HTTP multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the homepage
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

# Route to start real-time detection
@app.route('/start_detection', methods=['POST'])
def start_detection():
    global cap, is_detecting
    if is_detecting:
        return jsonify({"message": "Detection is already running."})

    # Access camera or video feed
    video_source = request.json.get('video_source', 0)  # Default to webcam
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        return jsonify({"message": "Could not open video source"}), 400

    is_detecting = True
    return jsonify({"message": "Detection started."})

# Route to shut down the server and show the prediction summary
@app.route('/shutdown', methods=['POST'])
def shutdown():
    global cap, is_detecting

    # Stop the detection
    is_detecting = False
    if cap is not None:
        cap.release()

    # Generate a summary of predictions
    total_predictions = sum(prediction_summary.values())
    summary = "Summary of Predictions:\n"
    for label, count in prediction_summary.items():
        percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
        summary += f"{label}: {count} frames ({percentage:.2f}%)\n"

    print("\n" + summary)  # Print summary to console

    # Shut down the Flask server
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

    return "Server shutting down...\n" + summary

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
