import numpy as np
import tables
import cv2
import os

CLASS_NAMES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "bus",
    4: "truck",
    5: "bicycle",
    6: "motorcycle",
    7: "train"
}

def clear_directory(directory):
    """Create directory if not exists or remove all existing files inside."""
    os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def save_event_stack(stack, index, output_dir, frame_size):
    """Save the 3D event stack as a colored image."""
    color_image = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    color_image[stack == 0] = [0, 0, 0]
    color_image[stack == 128] = [255, 255, 255]
    color_image[stack == 255] = [180, 210, 255]
    color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    save_path = os.path.join(output_dir, f'{index:06}.png')
    cv2.imwrite(save_path, color_image_bgr)

def process_event_data(events_file, timestamps_file, output_dir, frame_size, chunk_size):
    """Main logic for processing H5 event data into normalized event stacks."""
    clear_directory(output_dir)
    
    with tables.open_file(events_file, mode='r') as file:
        total_events = file.get_node('/events/x').shape[0]
        t_offset = file.get_node('/t_offset').read()
        timestamps = np.loadtxt(timestamps_file)

        # Initialize frame
        event_frame = np.full((frame_size[1], frame_size[0]), 128, dtype=np.uint8)

        last_ts_idx = 0
        for start in range(0, total_events, chunk_size):
            end = min(start + chunk_size, total_events)
            x = file.get_node('/events/x')[start:end].astype(np.uint16)
            y = file.get_node('/events/y')[start:end].astype(np.uint16)
            t = file.get_node('/events/t')[start:end].astype(np.uint32) + t_offset
            p = file.get_node('/events/p')[start:end].astype(np.uint8)

            chunk_ptr = 0
            for i in range(last_ts_idx, len(timestamps)):
                finished_stack = False 
                for n in range(chunk_ptr, len(t)):
                    if t[n] <= timestamps[i]:
                        curr_p = p[n]
                        if curr_p == 1:
                            event_frame[y[n], x[n]] = 255
                        else:
                            event_frame[y[n], x[n]] = 0

                    else:
                        # Save stack if not at the very beginning/end
                        if 0 < i < (len(timestamps) - 1):
                            save_event_stack(event_frame, i - 1, output_dir, frame_size)
                        
                        event_frame.fill(128)
                        chunk_ptr = n
                        finished_stack = True
                        break
                
                if not finished_stack:
                    last_ts_idx = i
                    break

def draw_bbox(image, class_id, x_center, y_center, width, height, img_width, img_height):
    """Draw a labeled bounding box around an object in an image."""
    x_center, y_center = int(x_center * img_width), int(y_center * img_height)
    width, height = int(width * img_width), int(height * img_height)

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # Draw bounding box (Orange/Bgr)
    color = (0, 165, 255)  
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Draw text background and text label
    class_name = CLASS_NAMES.get(class_id, "Unknown")
    text_color = (0, 0, 0)
    (text_width, text_height), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
    
    cv2.rectangle(
        image, 
        (x1, y1 - text_height - 3), 
        (x1 + text_width, y1 + 3), 
        (255, 255, 255), 
        -1
    )
    cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1)


def visualize_labels(label_visual_dir, images_folder, labels_folder, img_width, img_height):
    """Visualize labels by drawing bounding boxes on images and saving the results."""
    clear_directory(label_visual_dir)

    for image_name in os.listdir(images_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_folder, image_name)
            label_path = os.path.join(labels_folder, os.path.splitext(image_name)[0] + '.txt')

            if not os.path.exists(label_path):
                continue

            image = cv2.imread(image_path)
            with open(label_path, 'r') as file:
                for line in file:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    draw_bbox(
                        image, int(class_id), x_center, y_center, 
                        width, height, img_width, img_height
                    )

            output_path = os.path.join(label_visual_dir, image_name)
            cv2.imwrite(output_path, image)

def main():
    mode = 'train'
    route = 'zurich_city_09_e'
    frame_size = (640, 480)
    chunk_size = 10000

    # Paths setup
    events_file = f'/home/loweeminh/Downloads/DSEC-Det/{mode}/{route}/events/events.h5'
    timestamps_file = f'/home/loweeminh/Downloads/DSEC-Det/label/{mode}/{route}/timestamps.txt'
    images_folder = f'/home/loweeminh/RGB-E-Detection/visualization/event_frame/{route}'
    labels_folder = f'/home/loweeminh/RGB-E-Detection/dsec_datasets/events/{mode}/labels/{route}'
    output_dir = f'/home/loweeminh/RGB-E-Detection/visualization/with_labels/{route}'

    # Process event data and generate frames
    if os.path.exists(events_file) and os.path.exists(timestamps_file):
        print(f"Processing event data for route: {route}...")
        process_event_data(events_file, timestamps_file, images_folder, frame_size, chunk_size)
    else:
        print("Error: Event file or timestamp file not found.")
        return

    # Visualize bounding box labels onto generated frames
    if os.path.exists(images_folder) and os.path.exists(labels_folder):
        print("Visualizing labels...")
        visualize_labels(output_dir, images_folder, labels_folder, frame_size[0], frame_size[1])
        print("Done!")
    else:
        print("Error: Image folder or label folder not found.")

if __name__ == "__main__":
    main()