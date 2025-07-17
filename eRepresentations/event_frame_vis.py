import numpy as np
import tables
import cv2
import os

# Function to remove all files in the output directory
def Del_Old_Files(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Function to save the event stack as a colored image
def Save_Event_Stacks(event_stack, count, output_dir, frame_size):
    color_image = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    color_image[event_stack == 0] = [0, 0, 0]
    color_image[event_stack == 128] = [255, 255, 255]
    color_image[event_stack == 255] = [180, 210, 255]
    color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    save_path = os.path.join(output_dir, f'{count:06}.png')
    cv2.imwrite(save_path, color_image_bgr)

# Function to process event data, generate event stack, and save them
def Process_Event_Data(events_file, timestamps_file, output_dir, frame_size, chunk_size):
    Del_Old_Files(output_dir)
    
    with tables.open_file(events_file, mode='r') as file:
        total_events = file.get_node('/events/x').shape[0]
        t_o = file.get_node('/t_offset').read()
        timestamps = np.loadtxt(timestamps_file)

        event_stack = np.full((frame_size[1], frame_size[0]), 128, dtype=np.uint8)

        last = 0

        for start in range(0, total_events, chunk_size):
            end = min(start + chunk_size, total_events)

            x = file.get_node('/events/x')[start:end].astype(np.uint16)
            y = file.get_node('/events/y')[start:end].astype(np.uint16)
            t = file.get_node('/events/t')[start:end].astype(np.uint32)
            p = file.get_node('/events/p')[start:end].astype(np.uint8)

            t = t + t_o

            prev = 0

            for i in range(last, len(timestamps)):
                break_flg = False 
                for n in range(prev, len(t)):
                    if t[n] <= timestamps[i]:
                        if p[n] == 1:
                            event_stack[y[n], x[n]] = 255
                        else:
                            event_stack[y[n], x[n]] = 0
                    else:
                        # Skip the first and last event stacks
                        if i != 0 and i != (len(timestamps) - 1):
                            Save_Event_Stacks(event_stack, i - 1, output_dir, frame_size)
                        event_stack.fill(128)
                        prev = n
                        break_flg = True
                        break
                if not break_flg:
                    last = i 
                    break

    print("Processing complete.")

# Function to draw a labeled bounding box around an object in an image
def Draw_BBox(image, class_id, x_center, y_center, width, height, img_width, img_height):
    class_names = {
        0: "pedestrian",
        1: "rider",
        2: "car",
        3: "bus",
        4: "truck",
        5: "bicycle",
        6: "motorcycle",
        7: "train"
    }

    x_center, y_center = int(x_center * img_width), int(y_center * img_height)
    width, height = int(width * img_width), int(height * img_height)

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    color = (0, 165, 255)  
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    class_name = class_names.get(class_id, "Unknown")
    text_color = (0, 0, 0)
    (text_width, text_height), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
    
    rectangle_x1 = x1
    rectangle_y1 = y1 - text_height - 3
    rectangle_x2 = x1 + text_width
    rectangle_y2 = y1 + 3
    cv2.rectangle(image, (rectangle_x1, rectangle_y1), (rectangle_x2, rectangle_y2), (255, 255, 255), -1)
    
    cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1)

# Function to visualize labels by drawing bounding boxes on images and saving the results
def Visualize_Label(label_visual_dir, images_folder, labels_folder, img_width, img_height):
    Del_Old_Files(label_visual_dir)

    for image_name in os.listdir(images_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_folder, image_name)
            label_path = os.path.join(labels_folder, os.path.splitext(image_name)[0] + '.txt')

            image = cv2.imread(image_path)
            with open(label_path, 'r') as file:
                for line in file:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    Draw_BBox(image, int(class_id), x_center, y_center, width, height, img_width, img_height)

            output_path = os.path.join(label_visual_dir, image_name)
            cv2.imwrite(output_path, image)

# Main function
def main():
    # Process event data, generate event stacks, and save them
    Process_Event_Data(g_events_file, g_timestamps_file, g_output_dir, g_frame_size, g_chunk_size)

    # Visualize labels
    Visualize_Label(g_label_visual_dir, g_images_folder, g_labels_folder, g_frame_size[0], g_frame_size[1])

if __name__ == "__main__":
    g_mode = 'train'
    g_route = 'zurich_city_09_e'
    g_events_file = '../DSEC-Det/' + g_mode + '/' + g_route + '/events/events.h5'
    g_timestamps_file = '../DSEC-Det/label/' + g_mode + '/' + g_route + '/timestamps.txt'
    g_output_dir = '../Label-Visual/event_frame_vis/' + g_route
    g_label_visual_dir = '../Label-Visual/with_labels_gt/' + g_route
    g_images_folder = '../Label-Visual/event_frame_vis/' + g_route
    g_labels_folder = '../datasets/dsec_events/' + g_mode + '/labels/' + g_route
    g_frame_size = (640, 480) # (width, height)
    g_chunk_size = 10000 # split loaded data due to limited RAM

    main()