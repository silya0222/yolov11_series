# my_ultra/losses.py
from ultralytics.utils.loss import v8DetectionLoss


class MyDetectionLoss(v8DetectionLoss):
    def __call__(self, preds, batch):
        # 在这里做你想要的改动
        return super().__call__(preds, batch)
