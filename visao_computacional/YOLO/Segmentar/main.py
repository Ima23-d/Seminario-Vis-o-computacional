from ultralytics import YOLO
modelo = YOLO("yolov8n-seg.pt")
modelo.predict(source="YOLO/video/video.mp4", show=True)