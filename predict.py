from ultralytics import YOLO

model = YOLO("/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt")  # or wherever your best model is saved
results = model.predict(
    source="/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/NumPlateTest1.webp",
    conf=0.3,
    show=True,
    save=True,
    project="/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/DetectionResults",  # top-level folder
    name="ImageResults"           # subfolder where results are saved
)