import cv2
import mediapipe as mp
import joblib
from collections import deque

# Load models
alphabet_model = joblib.load("backend/psl_gesture_model.pkl")
sentence_model = joblib.load("backend/psl_sentence_model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Smoothing
recent_predictions = deque(maxlen=10)

# Start in alphabet mode
mode = "alphabet"

print("📷 Webcam started. Press 'a' for alphabet mode, 's' for sentence mode, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
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
            if mode == "alphabet":
                prediction = alphabet_model.predict([keypoints])[0]
            else:
                prediction = sentence_model.predict([keypoints])[0]
            recent_predictions.append(prediction)
        else:
            recent_predictions.append("")
    else:
        recent_predictions.append("")

    if recent_predictions:
        most_common = max(set(recent_predictions), key=recent_predictions.count)
        display_text = f"{mode.capitalize()} Mode: {most_common}"
        cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("PSL Live Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        mode = "alphabet"
        print("🔤 Switched to Alphabet Mode")
    elif key == ord('s'):
        mode = "sentence"
        print("🗨️ Switched to Sentence Mode")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
