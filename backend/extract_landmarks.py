import mediapipe as mp
import cv2
import os
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
data = []

def extract_landmarks_from_frames():
    frames_dir = "backend/PSL_Frames"
    output_csv = "backend/psl_gesture_dataset.csv"

    for label in os.listdir(frames_dir):
        label_dir = os.path.join(frames_dir, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            keypoints = []

            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    for lm in hand.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])

                if len(keypoints) in [63, 126]:
                    row = keypoints + [label]
                    data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved landmark data to {output_csv}")

if __name__ == "__main__":
    extract_landmarks_from_frames()
