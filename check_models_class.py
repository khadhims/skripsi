from ultralytics import YOLO

model = YOLO("./models/yolov8/v8-nano.pt")

print(model.names)