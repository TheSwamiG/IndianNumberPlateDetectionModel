from ultralytics import YOLO

# Load a model
model = YOLO("/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt")  # build a new model from scratch

# Use the model
results = model.train(data="/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/config.yaml", epochs= 50)  # train the model