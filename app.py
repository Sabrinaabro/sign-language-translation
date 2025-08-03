from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
import cv2
import pyttsx3
import joblib
import mediapipe as mp
from collections import deque
import threading

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")

# === Paths ===
VIDEO_DIR = os.path.abspath("backend/PSL Dataset/Sentences")
ALPHABET_MODEL_PATH = "backend/psl_gesture_model.pkl"
SENTENCE_MODEL_PATH = "backend/psl_sentence_model.pkl"

# === Load models ===
alphabet_model = joblib.load(ALPHABET_MODEL_PATH)
sentence_model = joblib.load(SENTENCE_MODEL_PATH)

# === Speech ===
engine = pyttsx3.init()

# === MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === Webcam ===
camera = cv2.VideoCapture(0)
recent_predictions = deque(maxlen=10)
current_mode = "alphabet"  # default mode

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    text = request.form.get("text", "").strip().lower()

    if not text:
        return jsonify({"error": "Text cannot be empty."}), 400

    video_filename = f"{text}.mp4"
    video_path = os.path.join(VIDEO_DIR, video_filename)

    if not os.path.exists(video_path):
        return jsonify({"error": f"No video found for: {text}"}), 404

    threading.Thread(target=speak_text, args=(text,)).start()
    return jsonify({"video_url": f"/videos/{video_filename}"})

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

@app.route("/videos/<path:filename>")
def serve_video(filename):
    return send_from_directory(VIDEO_DIR, filename)

@app.route("/webcam")
def webcam():
    global current_mode
    mode = request.args.get("mode", "alphabet")
    if mode in ["alphabet", "sentence"]:
        current_mode = mode
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global current_mode
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        keypoints = []

        if result.multi_hand_landmarks:
            hands_detected = result.multi_hand_landmarks

            for i in range(2):
                if i < len(hands_detected):
                    for lm in hands_detected[i].landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])
                else:
                    keypoints.extend([0.0] * 63)

            for hand_landmarks in hands_detected:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if len(keypoints) == 126:
                model = alphabet_model if current_mode == "alphabet" else sentence_model
                prediction = model.predict([keypoints])[0]
                recent_predictions.append(prediction)
            else:
                recent_predictions.append("")
        else:
            recent_predictions.append("")

        if recent_predictions:
            most_common = max(set(recent_predictions), key=recent_predictions.count)
            display_text = f"{current_mode.capitalize()} Mode: {most_common}"
            cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == "__main__":
    app.run(debug=True)
