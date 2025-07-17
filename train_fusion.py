import sys
import os

sys.path.append(os.path.abspath("ultralytics-main"))
from ultralytics import YOLO

# Paper "Adaptive Token Fusion Transformer for Multi-Modal Object Detection with RGB and Event Data"
def main(mode):
    if mode == "event":
        # event data single modality train
        model = YOLO("ultralytics-main/yolo11l_event.yaml")
        model.train(epochs=30, close_mosaic=0, patience=3, data="ultralytics-main/dsec_path.yaml", sensor="event", channel_event=3, plots=False)
    elif mode == "rgb":
        # rgb data single modality train
        model = YOLO("ultralytics-main/yolo11l_rgb.yaml")
        model.train(epochs=30, close_mosaic=0, patience=3, data="ultralytics-main/dsec_path.yaml", sensor="rgb", plots=False)
    elif mode == "fusion":
        # fusion model train
        model = YOLO("ultralytics-main/yolo11l_fusion.yaml")
        model.train(epochs=30, close_mosaic=0, patience=3, data="ultralytics-main/dsec_path.yaml", sensor="fusion", plots=False)
    
if __name__ == "__main__":
    main(mode="fusion")