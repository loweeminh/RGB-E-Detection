import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Adaptive Token Fusion Transformer Validation Script")
    
    parser.add_argument("--mode", type=str, default="fusion", choices=["rgb", "event", "fusion"], help="Validation mode")
    parser.add_argument("--data", type=str, default="dsec", choices=["pku", "dsec"], help="Dataset name")
    
    parser.add_argument("--weights", type=str, default=None, help="Path to specific weights")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--split", type=str, default=None, choices=["val", "test"], help="Dataset split to use")
    parser.add_argument("--plots", action="store_true", help="Enable validation plots")

    parser.add_argument("--channel_event", type=int, default=3, help="Event data channels")
    
    args = parser.parse_args()

    ROOT_DIR = "/home/loweeminh/RGB-E-Detection/ultralytics-main/"

    if args.split is None:
        if args.data == "pku":
            args.split = "test"
        elif args.data == "dsec":
            args.split = "val"
        else:
            args.split = None

    dataset_map = {
        "pku": os.path.join(ROOT_DIR, "pku_path.yaml"),
        "dsec": os.path.join(ROOT_DIR, "dsec_path.yaml")
    }
    data_path = dataset_map[args.data]

    if args.weights is None:
        suffix = f"_{args.data}"
        weight_map = {
            "event": os.path.join(ROOT_DIR, f"runs/train/event{suffix}/weights/best.pt"),
            "rgb": os.path.join(ROOT_DIR, f"runs/train/rgb{suffix}/weights/best.pt"),
            "fusion": os.path.join(ROOT_DIR, f"runs/train/ATFT{suffix}/weights/best.pt")
        }
        args.weights = weight_map[args.mode]

    model = YOLO(args.weights)
    
    model.val(
        data=data_path,
        sensor=args.mode,
        batch=args.batch,
        split=args.split,
        plots=args.plots,
        channel_event=args.channel_event
    )

if __name__ == "__main__":
    main()