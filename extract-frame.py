import cv2
import os
import uuid

def extract_frames(video_path, output_folder, interval_sec=None, interval_frames=None):
    """
    Extracts frames from a video at specific intervals.
    Saves them as PNGs to preserve quality.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video Info: {total_frames} frames, {fps} FPS, {duration:.2f} seconds")

    # Determine frame jump
    frame_jump = 1
    if interval_frames is not None:
        frame_jump = int(interval_frames)
    elif interval_sec is not None:
        frame_jump = int(interval_sec * fps)
    
    if frame_jump < 1:
        frame_jump = 1

    print(f"Extracting every {frame_jump} frames...")

    current_frame = 0
    count = 0
    
    while True:
        # Seek to the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        ret, frame = cap.read()
        if not ret:
            break

        # Save as PNG
        output_filename = f"{str(uuid.uuid4())[:8]}.png"
        output_path = os.path.join(output_folder, output_filename)
        
        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path} (Time: {current_frame/fps:.2f}s)")
        
        count += 1
        current_frame += frame_jump
        
        if current_frame >= total_frames:
            break

    print(f"Done! Extracted {count} frames to {output_folder}")
    cap.release()

# Configuration Variables
VIDEO_PATH = "D:/DATASET/20251211104916730_FY0213996_hcDownloadP_Camera-Pemorsian_6_video.MOV" # Ganti dengan path video Anda
OUTPUT_FOLDER = "D:/DATASET/2025-12-11(Test)" # Folder untuk menyimpan hasil
INTERVAL_SEC = 1 # Ekstrak setiap X detik (misal: 1.0 untuk setiap 1 detik)
INTERVAL_FRAMES = None # Ekstrak setiap X frame (jika INTERVAL_SEC None)

if __name__ == "__main__":
    extract_frames(VIDEO_PATH, OUTPUT_FOLDER, INTERVAL_SEC, INTERVAL_FRAMES)
