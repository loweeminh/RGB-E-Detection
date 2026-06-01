import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Adaptive Token Fusion Transformer Inference Script")
    parser.add_argument("--mode", type=str, default="fusion", choices=["rgb", "event", "fusion"], help="Inference mode")
    parser.add_argument("--data", type=str, default="dsec", choices=["pku", "dsec"], help="Dataset name")
    parser.add_argument("--weights", type=str, default=None, help="Path to specific weights")

    args = parser.parse_args()

    ROOT_DIR = "/home/loweeminh/RGB-E-Detection/ultralytics-main/"

    if args.weights is None:
        suffix = f"_{args.data}"
        weight_map = {
            "event": os.path.join(ROOT_DIR, f"runs/train/event{suffix}/weights/best.pt"),
            "rgb": os.path.join(ROOT_DIR, f"runs/train/rgb{suffix}/weights/best.pt"),
            "fusion": os.path.join(ROOT_DIR, f"runs/train/ATFT{suffix}/weights/best.pt")
        }
        args.weights = weight_map[args.mode]
        
    model = YOLO(args.weights)

    dataset_root = os.path.abspath(os.path.join(ROOT_DIR, "..", f"{args.data}_datasets"))
    splits = ["train", "val", "test"]

    for split in splits:
        sensor_folder = "events" if args.mode in ["event", "fusion"] else "rgb"
        images_dir = os.path.join(dataset_root, sensor_folder, split, "images")

        if not os.path.exists(images_dir):
            continue

        routes = [f for f in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, f))]
        
        for route in routes:
            img_path = os.path.join(images_dir, route)
            img_path2 = os.path.join(dataset_root, "rgb", split, "images", route) if args.mode == "fusion" else None

            results = model(
                source=img_path, 
                source2=img_path2,
                sensor=args.mode,
                save_txt=True, 
                stream=True, 
                project=f"runs/inference/{args.mode}_{args.data}",
                name=route,
                exist_ok=True
            )
            
            for result in results:
                result.save()

if __name__ == "__main__":
    main()