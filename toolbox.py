import numpy as np
import os
import shutil
from pathlib import Path

def clear_and_make_dir(path):
    """Recreate a clean directory."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def copy_files(src, dest):
    """Copy all files from src to dest."""
    os.makedirs(dest, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dest, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)

def process_images(src_folder, dest_folder):
    """Remove first/last images and rename remainder sequentially."""
    copy_files(src_folder, dest_folder)
    
    files = sorted([f for f in os.listdir(dest_folder) 
                    if f[:-4].isdigit() and len(f) == 10 and f.endswith('.png')])

    if len(files) >= 2:
        os.remove(os.path.join(dest_folder, files[0]))
        os.remove(os.path.join(dest_folder, files[-1]))

    remaining_files = sorted([f for f in os.listdir(dest_folder) 
                             if f[:-4].isdigit() and len(f) == 10 and f.endswith('.png')])

    for file in remaining_files:
        new_name = f"{int(file[:-4]) - 1:06d}.png"
        os.rename(os.path.join(dest_folder, file), os.path.join(dest_folder, new_name))

def reformat_to_yolo(class_id, x, y, w, h, img_w, img_h):
    """Convert bounding box to YOLO normalized format."""
    x_c = (x + w / 2) / img_w
    y_c = (y + h / 2) / img_h
    nw, nh = w / img_w, h / img_h
    
    x_c = max(0.0, min(x_c, 1.0))
    y_c = max(0.0, min(y_c, 1.0))
    nw = max(0.0, min(nw, 1.0))
    nh = max(0.0, min(nh, 1.0))
    
    return f"{int(class_id)} {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}"

def generate_labels(ts_path, track_path, out_dir, img_w, img_h):
    """Generate YOLO label files based on timestamps and tracking data."""
    clear_and_make_dir(out_dir)
    timestamps = np.loadtxt(ts_path)
    tracks = np.load(track_path)
    
    cursor = 0
    for t_idx in range(1, len(timestamps) - 1):
        out_file = os.path.join(out_dir, f"{t_idx - 1:06d}.txt")
        with open(out_file, 'w') as f:
            for i in range(cursor, len(tracks)):
                if tracks[i][0] < timestamps[t_idx]:
                    continue
                elif tracks[i][0] == timestamps[t_idx]:
                    label = reformat_to_yolo(tracks[i][5], tracks[i][1], tracks[i][2], 
                                            tracks[i][3], tracks[i][4], img_w, img_h)
                    f.write(label + '\n')
                else:
                    cursor = i
                    break

def main():
    # Configuration
    CONFIG = {
        "in_root": "/home/loweeminh/Downloads/DSEC-Det",
        "out_root": "dsec_datasets",
        "splits": ["train", "val"],
        "sensors": ["rgb", "events"],
        "img_size": (640, 480)
    }

    for split in CONFIG["splits"]:
        split_path = os.path.join(CONFIG["in_root"], split)
        if not os.path.exists(split_path):
            continue

        # Automatically get all subfolders (routes)
        routes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        print(f"--- Processing {split} split ({len(routes)} routes) ---")

        for route in routes:
            # 1. Image Processing (RGB)
            rgb_src = os.path.join(split_path, route, 'rgb')
            rgb_dest = os.path.join(CONFIG["out_root"], "rgb", split, "images", route)
            if os.path.exists(rgb_src):
                process_images(rgb_src, rgb_dest)

            # 2. Label Generation (For each sensor)
            ts_path = os.path.join(CONFIG["in_root"], "label", split, route, "timestamps.txt")
            track_path = os.path.join(CONFIG["in_root"], "label", split, route, "tracks.npy")

            if os.path.exists(ts_path) and os.path.exists(track_path):
                for sensor in CONFIG["sensors"]:
                    label_dest = os.path.join(CONFIG["out_root"], f"{sensor}", split, "labels", route)
                    generate_labels(ts_path, track_path, label_dest, *CONFIG["img_size"])
            
            print(f"Done: {route}")

    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()