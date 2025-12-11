import cv2
import time
import logging
from ultralytics import YOLO
import os

# --- Configuration ---
# Ganti path ini dengan path model yang sudah Anda training di Kaggle
# Contoh: 'runs/detect/train/weights/best.pt' atau path absolutnya.
# Jika file berada di folder yang sama, cukup tulis nama filenya.
MODEL_PATH = "D:/somba-trains-result/2025-12-11/train-v4/yolov8-train-v4.pt"  # Default contoh, ganti dengan model Anda

# Ganti path ini dengan path video yang ingin dites
# Gunakan 0 untuk webcam, atau string path ke file video (contoh: 'video.mp4')
VIDEO_PATH = "D:/DATASET/20251211103859757_FY0213996_hcDownloadP_Camera-Persiapan_4_video.MOV" # Default webcam, ganti dengan path video Anda

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference_log.txt"),
        logging.StreamHandler()
    ]
)

def main():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH) and MODEL_PATH != 'yolov8n.pt':
        logging.error(f"Model file not found: {MODEL_PATH}")
        return

    logging.info(f"Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # 2. Open Video Source
    logging.info(f"Opening video source: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        logging.error("Error: Could not open video source.")
        return

    # Get video properties for saving (optional, not requested but good practice)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    
    logging.info("Starting inference loop...")
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video stream or failed to read frame.")
            break

        frame_count += 1
        
        # 3. Predict & Measure Time
        start_time = time.time()
        
        # Perform inference on the frame
        # stream=True returns a generator, so we must iterate to trigger inference
        results = model.predict(frame, verbose=False, conf=0.7, stream=True, imgsz=1088, iou=0.4, classes=[2,3,4])
        
        for result in results:
            annotated_frame = result.plot()
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000 # Convert to milliseconds

        # 4. Log Time
        logging.info(f"Frame {frame_count}: Inference Time = {inference_time:.2f} ms")

        # 5. Visualize Results
        # Visualize the results on the frame
        # annotated_frame is already set in the loop

        # Display inference time on the frame itself
        cv2.putText(annotated_frame, f"Inference Time: {inference_time:.2f} ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        display_frame = cv2.resize(annotated_frame, (0, 0), fx=5/6, fy=5/6)
        cv2.imshow("YOLO Detection", display_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("User interrupted processing.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
