import os
import cv2
import joblib
import mediapipe as mp
from collections import Counter

# Load the trained model
model = joblib.load("backend/psl_gesture_model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# Path to alphabet videos
alphabet_folder = "backend/psl dataset/alphabets"

# Test all videos in the folder
for file in os.listdir(alphabet_folder):
    if file.endswith(".mp4"):
        label = os.path.splitext(file)[0]  # e.g., A, B, C
        video_path = os.path.join(alphabet_folder, file)
        cap = cv2.VideoCapture(video_path)
        predictions = []
        frame_id = 0

        print(f"\n🔤 Testing video: {file} (Expected: {label})")

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            frame_id += 1
            if frame_id % 5 != 0:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            keypoints = []
            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    for lm in hand.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])

                if len(keypoints) == 126:  # Expecting 2 hands
                    prediction = model.predict([keypoints])[0]
                    predictions.append(prediction)

        cap.release()

        # Summarize
        if predictions:
            summary = Counter(predictions)
            final = summary.most_common(1)[0][0]
            print(f"✅ Predicted: {final} | Expected: {label} | Match: {'✔️' if final == label else '❌'}")
        else:
            print("⚠️ No hands detected or insufficient features.")

