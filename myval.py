from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"D:\deeplearning\ultralytics-8.3.232\results\yolo11n\weights\best.pt")
    model.val(split="test")
