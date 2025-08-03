import cv2
import mediapipe as mp
import joblib
import pandas as pd

# Load trained model
model = joblib.load("backend/psl_sentence_model.pkl")
video_path = "backend/PSL Dataset/Sentences/Goodbye.mp4"

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

cap = cv2.VideoCapture(video_path)
landmarks_all = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

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

        if len(keypoints) == 126:
            landmarks_all.append(keypoints)

cap.release()

if landmarks_all:
    avg_features = pd.DataFrame(landmarks_all).mean().tolist()
    prediction = model.predict([avg_features])[0]
    print(f"✅ Predicted sentence: {prediction}")
else:
    print("⚠️ No hands detected.")
