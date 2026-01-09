import re
from collections import defaultdict
import os

def calculate_average_inference(log_path):
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    model_inference_times = defaultdict(list)
    current_model = "Unknown Model"

    print(f"Reading log file: {log_path}...")
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Check for model loading
            # Example: 2025-12-11 13:37:12,796 - INFO - Loading model from: D:/somba-trains-result/2025-12-11/train-v4/yolov8-train-v4.pt
            model_match = re.search(r'Loading model from: (.+)', line)
            if model_match:
                current_model = model_match.group(1).strip()
                continue

            # Check for inference time
            # Example: 2025-12-11 13:37:13,342 - INFO - Frame 1: Inference Time = 350.01 ms
            time_match = re.search(r'Inference Time = (\d+\.\d+) ms', line)
            if time_match:
                inference_time = float(time_match.group(1))
                model_inference_times[current_model].append(inference_time)

    # Calculate and print averages
    print("\n" + "="*100)
    print(f"{'Model Name':<60} | {'Avg Time (ms)':<15} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Frames':<5}")
    print("="*100)
    
    for model, times in model_inference_times.items():
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            # Shorten model path for display if it's too long
            display_name = model
            if len(display_name) > 58:
                display_name = "..." + display_name[-55:]
                
            print(f"{display_name:<60} | {avg_time:<15.2f} | {min_time:<10.2f} | {max_time:<10.2f} | {len(times):<5}")
    print("="*100 + "\n")

if __name__ == "__main__":
    calculate_average_inference('inference_log.txt')
