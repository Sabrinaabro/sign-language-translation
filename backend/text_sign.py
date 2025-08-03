import cv2
import os
import time
import tkinter as tk
from tkinter import messagebox

# Set video folders
sentence_dir = "backend/PSL Dataset/Sentences"
alphabet_dir = "backend/PSL Dataset/Alphabets"

def play_video(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Sign Language Playback", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def text_to_sign(input_text):
    input_text = input_text.strip().lower()
    sentence_path = os.path.join(sentence_dir, f"{input_text.capitalize()}.mp4")

    if os.path.exists(sentence_path):
        play_video(sentence_path)
    else:
        for char in input_text:
            if char == " ":
                time.sleep(0.5)
                continue
            video_file = os.path.join(alphabet_dir, f"{char.upper()}.mp4")
            if os.path.exists(video_file):
                play_video(video_file)
                time.sleep(0.2)
            else:
                print(f"⚠️ Missing video for letter: {char.upper()}")

def on_translate():
    text = entry.get()
    if not text:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return
    text_to_sign(text)

# Tkinter GUI setup
root = tk.Tk()
root.title("Text to PSL Translator")
root.geometry("400x200")

label = tk.Label(root, text="Enter text to translate to sign language:", font=("Arial", 12))
label.pack(pady=10)

entry = tk.Entry(root, font=("Arial", 14), width=30)
entry.pack(pady=5)

btn = tk.Button(root, text="Translate", font=("Arial", 12), command=on_translate)
btn.pack(pady=15)

root.mainloop()
