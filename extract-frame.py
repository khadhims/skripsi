import cv2
import argparse
import os

def extract_frame(video_path, output_path, time_sec=None, frame_num=None):
    """
    Extracts a frame from a video at a specific time or frame number.
    Saves it as a PNG to preserve quality.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    target_frame = 0
    if frame_num is not None:
        target_frame = frame_num
    elif time_sec is not None:
        target_frame = int(time_sec * fps)
    
    if target_frame >= total_frames:
        print(f"Error: Target frame {target_frame} exceeds total frames {total_frames}.")
        cap.release()
        return

    # Seek to the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    
    # Read the frame
    ret, frame = cap.read()
    if ret:
        # Save as PNG (lossless)
        # Ensure output path ends with .png
        if not output_path.lower().endswith('.png'):
            output_path += '.png'
            
        cv2.imwrite(output_path, frame)
        print(f"Success: Frame {target_frame} extracted to {output_path}")
        print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("Error: Could not read frame.")

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a frame from a video without losing resolution.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output", required=True, help="Path to the output image file (e.g., output.png)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--time", type=float, help="Time in seconds to extract frame from")
    group.add_argument("--frame", type=int, help="Frame number to extract")

    args = parser.parse_args()

    extract_frame(args.video_path, args.output, args.time, args.frame)
