import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Adaptive Token Fusion Transformer Training Script")
    
    parser.add_argument("--mode", type=str, default="fusion", choices=["rgb", "event", "fusion"], help="Training mode")
    parser.add_argument("--data", type=str, default="dsec", choices=["pku", "dsec"], help="Dataset name")
    
    parser.add_argument("--imgsz", type=int, default=None, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--plots", action="store_true", help="Enable validation plots")

    parser.add_argument("--channel_event", type=int, default=3, help="Event data channels")
    parser.add_argument("--close_mosaic", type=int, default=0, help="Close mosaic at last N epochs")
    
    args = parser.parse_args()

    ROOT_DIR = "/home/loweeminh/RGB-E-Detection/ultralytics-main/"

    if args.imgsz is None:
        if args.data == "pku":
            args.imgsz = 352
        elif args.data == "dsec":
            args.imgsz = 640
        else:
            args.imgsz = None

    dataset_map = {
        "pku": os.path.join(ROOT_DIR, "pku_path.yaml"),
        "dsec": os.path.join(ROOT_DIR, "dsec_path.yaml")
    }
    data_path = dataset_map[args.data]

    model_path = os.path.join(ROOT_DIR, f"yolo11l_{args.mode}.yaml")
    model = YOLO(model_path)

    if args.mode == "fusion":
        folder_name = f"ATFT_{args.data}"
    else:
        folder_name = f"{args.mode}_{args.data}"
    
    model.train(
        data=data_path,
        sensor=args.mode,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        plots=args.plots,
        channel_event=args.channel_event,
        close_mosaic=args.close_mosaic,
        project="runs/train",
        name=folder_name,
        exist_ok=True
    )

if __name__ == "__main__":
    main()