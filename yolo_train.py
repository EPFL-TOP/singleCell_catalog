from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a pretrained model

results = model.train(data="C:/Users/helsens/software/singleCell_catalog/data.yaml", epochs=100, imgsz=512)