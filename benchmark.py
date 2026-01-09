import cv2
import time
import psutil
import torch
import numpy as np
from tracker import ObjectDetection

def run_benchmark(video_path, max_frames=200):
    print(f"Starting benchmark on {video_path}...")
    print(f"Frames to process: {max_frames}")
    
    # Initialize Detector
    detector = ObjectDetection(capture_index=video_path)
    cap = cv2.VideoCapture(video_path)
    tracker = detector.tracker
    
    # Metrics
    cpu_usages = []
    mem_usages = []
    frame_times = []
    
    # Warmup
    if hasattr(tracker, 'model') and hasattr(tracker.model, 'warmup'):
        tracker.model.warmup()

    frame_count = 0
    process = psutil.Process()
    
    outputs = [None]
    
    print("Benchmarking started...")
    start_total = time.perf_counter()
    
    while cap.isOpened() and frame_count < max_frames:
        start_frame = time.perf_counter()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference & Tracking
        results = detector.predict(frame)
        
        for result in results:
            xywhs = result.boxes.xywh.cpu()
            confs = result.boxes.conf.cpu()
            clss = result.boxes.cls.cpu()
            outputs[0] = tracker.update(xywhs, confs, clss, frame)
            
        # No Drawing, No Imshow - Pure Processing
        
        # Record Metrics
        frame_time = time.perf_counter() - start_frame
        frame_times.append(frame_time)
        cpu_usages.append(process.cpu_percent())
        mem_usages.append(process.memory_info().rss / 1024 / 1024) # MB
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    end_total = time.perf_counter()
    
    avg_fps = frame_count / (end_total - start_total)
    avg_img_process_time = np.mean(frame_times) * 1000 # ms
    avg_cpu = np.mean(cpu_usages)
    max_mem = np.max(mem_usages)
    avg_mem = np.mean(mem_usages)
    
    print("\n" + "="*40)
    print("       BENCHMARK RESULTS       ")
    print("="*40)
    print(f"Total Frames    : {frame_count}")
    print(f"Total Time      : {end_total - start_total:.2f} s")
    print(f"Average FPS     : {avg_fps:.2f}")
    print(f"Avg Frame Time  : {avg_img_process_time:.2f} ms")
    print("-" * 40)
    print(f"Avg CPU Usage   : {avg_cpu:.2f}% (Process Only)")
    print(f"Max RAM Usage   : {max_mem:.2f} MB")
    print(f"Avg RAM Usage   : {avg_mem:.2f} MB")
    print("="*40)

if __name__ == "__main__":
    # Use the same video file as in tracker.py
    VIDEO_PATH = "D:/DATASET/20251211104916730_FY0213996_hcDownloadP_Camera-Pemorsian_6_video.MOV"
    run_benchmark(VIDEO_PATH)
