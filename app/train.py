from ultralytics import YOLO

model = YOLO("yolov26n.pt")

model.train(
    data="/Users/ermia/Desktop/RoboCup2026/yolo/thr_project/config.yaml",
    epochs=40,
    patience=12,
    imgsz=320,
    batch=8,
    device="mps",
    optimizer="AdamW",
    lr0=0.003,
    hsv_s=0.7,
    hsv_v=0.6,
    mosaic=0.8,
    mixup=0.10,
    close_mosaic=10,
)
