import torch
import numpy as np
import cv2
from time import perf_counter
from ultralytics import YOLO
import os
import yaml
from easydict import EasyDict as edict
from pathlib import Path

import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("inference_log.txt"),
        logging.StreamHandler()
    ]
)

import supervision as sv
from strongsort.strong_sort import StrongSORT
from strongsort.utils.parser import YamlParser

SAVE_VIDEO = True
CONFIRMATION_TIME = 0.8 # Detik

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device: ", self.device)
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True

        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT, thickness=3)
        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
        
        # Dictionary to store active tracks: {track_id: {'start_time': float, 'label': str, 'conf': float}}
        self.active_tracks = {}
        
        reid_weights = Path("strongsort/deep/checkpoint/osnet_x0_25_msmt17.pt")

        tracker_config = "strongsort/configs/strong_sort.yaml"
        cfg = YamlParser()
        cfg.merge_from_file(tracker_config)

        self.tracker = StrongSORT(
            reid_weights,
            torch.device(self.device),
            False, 
            max_dist=cfg.STRONGSORT.MAX_DIST,
            max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE, 
            max_age=cfg.STRONGSORT.MAX_AGE,
            n_init=cfg.STRONGSORT.N_INIT,
            nn_budget=cfg.STRONGSORT.NN_BUDGET,
            mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
            ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
        )
    
    def load_model(self):
        model = YOLO("./models/yolov8/v8-nano.pt")
        model.fuse()
        
        return model

    def predict(self, frame): 
        results = self.model.predict(frame, conf=0.3, imgsz=1088, verbose=False, iou=0.7)

        return results

    def draw_results(self, frame, detections):
        # Generate labels with Tracker ID
        self.labels = [
            f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]} {confidence:.2f}"
            for confidence, class_id, tracker_id
            in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]

        # Annotate frame
        frame = self.box_annotator.annotate(
            scene=frame, 
            detections=detections
        )
        
        # Use black text for better visibility
        frame = self.label_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=self.labels
        )
            
        return frame, detections.xyxy

    def save_detection_to_db(self, track_id, label, conf):
        # Placeholder for DB Logic
        logging.info(f"========> [REPORTED TO DB] ID {track_id} ({label}) Conf: {conf:.2f} <========")

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        frame_id = 0
        pred_file = open("pred.txt", "w")
        print("Generating pred.txt...")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if SAVE_VIDEO:
            outputvid = cv2.VideoWriter('result_tracking_3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        # setup pelacakan
        tracker = self.tracker
        
        # if tracker menggunakan model kemudian warmup
        if hasattr(tracker, 'model'):
            if hasattr(tracker.model, 'warmup'):
                tracker.model.warmup()

        outputs = [None]
        curr_frames, prev_frames = None, None
        
        while True:
            start_time = perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1

            results = self.predict(frame)

            num_det = sum(len(r.boxes) for r in results)
            print(f"Frame {frame_id} - YOLO detections: {num_det}")

            # Update tracker
            for result in results:
                xywhs = result.boxes.xywh.cpu()
                confs = result.boxes.conf.cpu()
                clss = result.boxes.cls.cpu()
                outputs[0] = tracker.update(xywhs, confs, clss, frame)

                print(f"Frame {frame_id} - Tracker outputs: {0 if outputs[0] is None else len(outputs[0])}")

            
            # Prepare tracked detections for visualization
            if outputs[0] is not None and len(outputs[0]) > 0:
                # outputs[0] structure: x1, y1, x2, y2, track_id, class_id, conf
                output_array = outputs[0]

                # ===============================
                # TAMBAHAN: Save to pred.txt (AGNOSTIC)
                # ===============================
                for row in output_array:
                    # format: x1, y1, x2, y2, track_id, class_id, conf
                    x1, y1, x2, y2, track_id, class_id, conf = row

                    x = float(x1)
                    y = float(y1)
                    w = float(x2 - x1)
                    h = float(y2 - y1)

                    # AGNOSTIC MODE → semua class = 1
                    mapped_class = 1

                    pred_file.write(
                        f"{frame_id},{int(track_id)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{float(conf):.4f},{mapped_class},1\n"
                    )

                tracked_detections = sv.Detections(
                    xyxy=output_array[:, 0:4],
                    confidence=output_array[:, 6],
                    class_id=output_array[:, 5].astype(int),
                    tracker_id=output_array[:, 4].astype(int)
                )
                frame, _ = self.draw_results(frame, tracked_detections)
                
                # --- LOGGING LOGIC ---
                current_time = time.time()
                current_ids = set(output_array[:, 4].astype(int))
                
                # Check for new tracks
                for i, track_id in enumerate(output_array[:, 4].astype(int)):
                    if track_id not in self.active_tracks:
                        cls_id = int(output_array[i, 5])
                        conf = output_array[i, 6]
                        label = self.CLASS_NAMES_DICT[cls_id]
                        
                        # New track Init
                        self.active_tracks[track_id] = {
                            'start_time': current_time,
                            'label': label,
                            'conf': conf,
                            'frames_seen': 1,
                            'reported': False
                        }
                        logging.info(f"New object detecting: ID {track_id} ({label}) Conf: {conf:.2f}")

                    else:
                        # Existing track update
                        self.active_tracks[track_id]['frames_seen'] += 1
                        
                        # Threshold Check
                        confirmation_frames = int(fps * CONFIRMATION_TIME)
                        if self.active_tracks[track_id]['frames_seen'] >= confirmation_frames and not self.active_tracks[track_id]['reported']:
                            # Trigger Database Report
                            self.save_detection_to_db(track_id, self.active_tracks[track_id]['label'], self.active_tracks[track_id]['conf'])
                            self.active_tracks[track_id]['reported'] = True

                # Check for lost tracks
                lost_ids = set(self.active_tracks.keys()) - current_ids
                for track_id in lost_ids:
                    duration = current_time - self.active_tracks[track_id]['start_time']
                    label = self.active_tracks[track_id]['label']
                    logging.info(f"Object lost: ID {track_id} ({label}) - Duration: {duration:.2f}s")
                    del self.active_tracks[track_id] # Remove from active tracking
            else:
                 # If no detections at all, check if we need to close any existing tracks
                 # (Optional: or keep them for a few frames? For now, let's assume if no output, they are lost)
                 current_time = time.time()
                 lost_ids = list(self.active_tracks.keys())
                 for track_id in lost_ids:
                    duration = current_time - self.active_tracks[track_id]['start_time']
                    label = self.active_tracks[track_id]['label']
                    logging.info(f"Object lost: ID {track_id} ({label}) - Duration: {duration:.2f}s")
                    del self.active_tracks[track_id]



            end_time = perf_counter()
            fps = 1/np.round(end_time - start_time, 2)
            # Resize for display (Canvas diperkecil 50%)
            display_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.putText(display_frame, f"FPS : {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("YOLOv8 detection", display_frame)

            if SAVE_VIDEO:
                outputvid.write(frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        if SAVE_VIDEO:
            outputvid.release()

        pred_file.close()
        print("Finished. pred.txt saved.")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetection(capture_index="./dataset/videos/vid_1.mp4")
    detector()