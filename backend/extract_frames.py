import cv2
import os

def extract_frames_from_alphabet_videos():
    dataset_dir = "backend/PSL Dataset/Alphabets"
    output_dir = "backend/PSL_Frames"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(dataset_dir):
        if not filename.endswith(".mp4"):
            continue

        label = os.path.splitext(filename)[0]
        video_path = os.path.join(dataset_dir, filename)
        save_path = os.path.join(output_dir, label)
        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_num = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            if frame_num % 5 == 0:
                cv2.imwrite(f"{save_path}/{label}_{frame_num}.jpg", frame)
            frame_num += 1
        cap.release()

    print("✅ Frames extracted.")

if __name__ == "__main__":
    extract_frames_from_alphabet_videos()
