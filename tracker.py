import torch
import numpy as np
import cv2
from time import perf_counter
from ultralytics import YOLO
import os
import yaml
from easydict import EasyDict as edict
from pathlib import Path

import supervision as sv
from strongsort.strong_sort import StrongSORT
from strongsort.utils.parser import YamlParser

SAVE_VIDEO = True

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
        model = YOLO("D:/somba-trains-result/2025-12-11/train-v4/yolov8-train-v4.pt")
        model.fuse()
        
        return model

    def predict(self, frame): 
        results = self.model.predict(frame, conf=0.4, imgsz=960, classes=[2,3,4])

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

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if SAVE_VIDEO:
            outputvid = cv2.VideoWriter('result_tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
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
            
            assert ret
            results = self.predict(frame)

            # Update tracker
            for result in results:
                xywhs = result.boxes.xywh.cpu()
                confs = result.boxes.conf.cpu()
                clss = result.boxes.cls.cpu()
                outputs[0] = tracker.update(xywhs, confs, clss, frame)
            
            # Prepare tracked detections for visualization
            if outputs[0] is not None and len(outputs[0]) > 0:
                # outputs[0] structure: x1, y1, x2, y2, track_id, class_id, conf
                output_array = outputs[0]
                tracked_detections = sv.Detections(
                    xyxy=output_array[:, 0:4],
                    confidence=output_array[:, 6],
                    class_id=output_array[:, 5].astype(int),
                    tracker_id=output_array[:, 4].astype(int)
                )
                frame, _ = self.draw_results(frame, tracked_detections)
            else:
                 # Fallback to raw detections if no tracks yet (optional, or just show empty)
                 # For consistency, better to show nothing or raw results. 
                 # Let's show raw results if tracker is empty to avoid blank screen init
                 pass # self.draw_results(frame, results) - actually draw_results expects Detections object now? 
                 # Wait, draw_results currently expects 'results' (YOLO object). I need to change draw_results signature too.


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
        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index="D:/DATASET/20251211103859757_FY0213996_hcDownloadP_Camera-Persiapan_4_video.MOV")
detector()