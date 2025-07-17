import sys
import os

sys.path.append(os.path.abspath("ultralytics-main"))
from ultralytics import YOLO

# Paper "Adaptive Token Fusion Transformer for Multi-Modal Object Detection with RGB and Event Data"
def main(mode):
    if mode == "event":
        # event data single modality validation
        model = YOLO("runs/detect/event/weights/best.pt")
        model.val(data="ultralytics-main/dsec_path.yaml", sensor="event", channel_event=3, plots=False)
    elif mode == "rgb":
        # rgb data single modality validation
        model = YOLO("runs/detect/rgb/weights/best.pt")
        model.val(data="ultralytics-main/dsec_path.yaml", sensor="rgb", plots=False)
    elif mode == "fusion":
        # fusion model validation
        model = YOLO("runs/detect/ATFT/weights/best.pt")
        model.val(data="ultralytics-main/dsec_path.yaml", sensor="fusion", plots=False)

if __name__ == "__main__":
    main(mode="fusion")