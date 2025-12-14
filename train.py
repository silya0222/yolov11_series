from ultralytics import YOLO

ms = [
    "yolo11n", "yolo11l", "yolo11m","yolo11s",
]

if __name__ == "__main__":
    for m in ms:
        # 注意：下面这两段代码必须缩进，表示它们在 for 循环里面
        model = YOLO(m + ".pt")
        model.train(
            data=r"km.yaml",
            epochs=100,
            imgsz=640,  # 越小训练越快
            batch=8,
            cache="ram",
            workers=1,
            project="results",
            name=m,
        )