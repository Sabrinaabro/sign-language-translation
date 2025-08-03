import os
import cv2
import mediapipe as mp
import pandas as pd

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# Folder paths
video_dir = "backend/PSL Dataset/Sentences"
output_csv = "backend/psl_sentence_dataset.csv"

data = []

def extract_features_from_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    collected_frames = 0
    all_keypoints = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count % 5 == 0:  # Sample every 5th frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)
            keypoints = []

            if result.multi_hand_landmarks:
                hands_detected = result.multi_hand_landmarks
                for i in range(2):  # Always enforce 2-hand space
                    if i < len(hands_detected):
                        for lm in hands_detected[i].landmark:
                            keypoints.extend([lm.x, lm.y, lm.z])
                    else:
                        keypoints.extend([0.0] * 63)  # Pad missing hand

            if len(keypoints) == 126:
                all_keypoints.append(keypoints)
                collected_frames += 1

        frame_count += 1

    cap.release()

    if collected_frames > 0:
        avg_features = [sum(f[i] for f in all_keypoints) / collected_frames for i in range(len(all_keypoints[0]))]
        data.append(avg_features + [label])
        print(f"✅ Processed: {os.path.basename(video_path)} | Frames: {collected_frames}")
    else:
        print(f"⚠️ No valid frames found in: {os.path.basename(video_path)}")

def main():
    for filename in os.listdir(video_dir):
        if not filename.endswith(".mp4"):
            continue
        label = os.path.splitext(filename)[0]
        video_path = os.path.join(video_dir, filename)
        extract_features_from_video(video_path, label)

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"📁 Saved dataset to: {output_csv}")
    else:
        print("❌ No data extracted. Please check your video files.")

if __name__ == "__main__":
    main()
