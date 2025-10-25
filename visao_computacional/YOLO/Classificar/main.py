from ultralytics import YOLO
modelo = YOLO("yolo11n-cls")
modelo.predict(source="YOLO/video/video.mp4", show=True)