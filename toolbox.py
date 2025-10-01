import numpy as np
import os
import shutil

# Function to copy files from the source folder to the destination folder
def Copy_Files(src_folder, dest_folder):
    if not os.path.exists(src_folder):
        print(f"Source folder '{src_folder}' does not exist.")
        return
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    for filename in os.listdir(src_folder):
        src_file = os.path.join(src_folder, filename)
        dest_file = os.path.join(dest_folder, filename)
        
        if os.path.isfile(src_file):
            shutil.copy(src_file, dest_file)

# Function to organize and rename image files by removing the first and last images and renaming the rest sequentially
def Organize_Files(folder_path, rename_folder_path):
    Del_Old_Files(rename_folder_path)
    Copy_Files(folder_path, rename_folder_path)
    files = os.listdir(rename_folder_path)   
    files = [f for f in files if f[:-4].isdigit() and len(f) == 10 and f.endswith('.png')]  
    files.sort()  

    if len(files) >= 2:
        first_image = files[0]
        last_image = files[-1]
        os.remove(os.path.join(rename_folder_path, first_image))
        os.remove(os.path.join(rename_folder_path, last_image))

    files = os.listdir(rename_folder_path)
    files = [f for f in files if f[:-4].isdigit() and len(f) == 10 and f.endswith('.png')]  
    files.sort()  

    for file in files:
        old_number = int(file[:-4])  
        new_number = old_number - 1       
        new_name = f'{new_number:06d}.png' 
        old_file_path = os.path.join(rename_folder_path, file)
        new_file_path = os.path.join(rename_folder_path, new_name)
        os.rename(old_file_path, new_file_path)  

# Function to remove all files in the output directory
def Del_Old_Files(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Function to reformat label data to YOLO format
def Reformat_Label(class_id, x, y, w, h, img_width, img_height):
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height

    x_center = min(max(x_center, 0), 1)
    y_center = min(max(y_center, 0), 1)
    width = min(max(width, 0), 1)
    height = min(max(height, 0), 1)

    return f"{class_id} {x_center} {y_center} {width} {height}"

# Function to generate label files from timestamps and tracking data, saving them to the specified directory
def Generate_Label_Files(timestamps_file, dsec_tracks_file, label_output_dir, img_width, img_height):
    Del_Old_Files(label_output_dir)

    timestamps = np.loadtxt(timestamps_file) 
    dsec_tracks = np.load(dsec_tracks_file)

    prev = 0

    for t in range(len(timestamps)):
        if t!= 0 and t != (len(timestamps) - 1):
            output_file = os.path.join(label_output_dir, f"{t - 1:06}.txt")
            with open(output_file, 'w') as file:
                for i in range(prev, dsec_tracks.shape[0]):
                    if dsec_tracks[i][0] < timestamps[t]:
                        continue
                    elif dsec_tracks[i][0] == timestamps[t]:
                        yolo_label = Reformat_Label(dsec_tracks[i][5], dsec_tracks[i][1], dsec_tracks[i][2], dsec_tracks[i][3], dsec_tracks[i][4], img_width, img_height)
                        file.write(yolo_label + '\n')
                    else:
                        prev = i
                        break

# Function to rename files in a sequential folders with sequential numbers for YOLO dataset format
def Rename_Yolo_Datasets(base_path, folder_name, start_index):
    folder_path = os.path.join(base_path, folder_name)
    files = sorted(os.listdir(folder_path))

    num_files = len(files)
    for idx, file_name in enumerate(reversed(files)):
        reversed_idx = num_files - 1 - idx
        file_extension = os.path.splitext(file_name)[1]
        new_name = f"{start_index + reversed_idx:06d}{file_extension}"
        
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

    return start_index + len(files)
   
# Main function
def main():
    # Organize and rename rgb image files
    print("Organizing and renaming image files... This may take a moment.")
    for route in g_folder_names_train:
        folder_path = 'DSEC-Det/train/' + route + '/rgb'
        rename_folder_path = 'datasets/dsec_rgb/train/' + '/images/' + route
        Organize_Files(folder_path, rename_folder_path)

    for route in g_folder_names_val:
        folder_path = 'DSEC-Det/val/' + route + '/rgb'
        rename_folder_path = 'datasets/dsec_rgb/val/' + '/images/' + route
        Organize_Files(folder_path, rename_folder_path)

    # Generate label files
    print("Generating label files ...")
    for sensor in g_sensors:
        for route in g_folder_names_train:
            timestamps_file = 'DSEC-Det/label/train/' + route + '/timestamps.txt'
            dsec_tracks_file = 'DSEC-Det/label/train/' + route + '/tracks.npy'
            label_output_dir = 'datasets/dsec_' + sensor + '/train/labels/' + route
            Generate_Label_Files(timestamps_file, dsec_tracks_file, label_output_dir, g_img_width, g_img_height)

        for route in g_folder_names_val:
            timestamps_file = 'DSEC-Det/label/val/' + route + '/timestamps.txt'
            dsec_tracks_file = 'DSEC-Det/label/val/' + route + '/tracks.npy'
            label_output_dir = 'datasets/dsec_' + sensor + '/val/labels/' + route
            Generate_Label_Files(timestamps_file, dsec_tracks_file, label_output_dir, g_img_width, g_img_height)

    # Rename files in a sequential folders with sequential numbers for YOLO dataset format
    print("Renaming files with sequential numbers for YOLO dataset format ...")
    for sensor in g_sensors:
        start_index_0 = 0
        start_index_1 = 0
        for folder_name in g_folder_names_train:
            base_path_images = 'datasets/dsec_' + sensor + '/train/images'
            base_path_labels = 'datasets/dsec_' + sensor + '/train/labels'
            start_index_0 = Rename_Yolo_Datasets(base_path_images, folder_name, start_index_0)
            start_index_1 = Rename_Yolo_Datasets(base_path_labels, folder_name, start_index_1)

        for folder_name in g_folder_names_val:
            base_path_images = 'datasets/dsec_' + sensor + '/val/images'
            base_path_labels = 'datasets/dsec_' + sensor + '/val/labels'
            start_index_0 = Rename_Yolo_Datasets(base_path_images, folder_name, start_index_0)
            start_index_1 = Rename_Yolo_Datasets(base_path_labels, folder_name, start_index_1)

    print("Processing complete!")

if __name__ == "__main__":
    g_sensors = ['rgb', 'events']

    g_img_width = 640
    g_img_height = 480

    g_folder_names_train = ['interlaken_00_c', 'interlaken_00_d', 'interlaken_00_e', 'interlaken_00_f', 'interlaken_00_g', 'thun_00_a',
                            'zurich_city_00_a', 'zurich_city_00_b', 'zurich_city_01_a', 'zurich_city_01_b', 'zurich_city_01_c', 
                            'zurich_city_01_d', 'zurich_city_01_e', 'zurich_city_01_f', 'zurich_city_02_a', 'zurich_city_02_b', 
                            'zurich_city_02_c', 'zurich_city_02_d', 'zurich_city_02_e', 'zurich_city_03_a', 'zurich_city_04_a', 
                            'zurich_city_04_b', 'zurich_city_04_c', 'zurich_city_04_d', 'zurich_city_04_e', 'zurich_city_04_f', 
                            'zurich_city_05_a', 'zurich_city_05_b', 'zurich_city_06_a', 'zurich_city_07_a', 'zurich_city_08_a', 
                            'zurich_city_09_a', 'zurich_city_09_b', 'zurich_city_09_c', 'zurich_city_09_d', 'zurich_city_09_e', 
                            'zurich_city_10_a', 'zurich_city_10_b', 'zurich_city_11_a', 'zurich_city_11_b', 'zurich_city_11_c', 
                            'zurich_city_16_a', 'zurich_city_17_a', 'zurich_city_18_a', 'zurich_city_19_a', 'zurich_city_20_a', 
                            'zurich_city_21_a']
    
    g_folder_names_val = ['interlaken_00_a', 'interlaken_00_b', 'interlaken_01_a', 'thun_01_a', 'thun_01_b', 'thun_02_a', 
                          'zurich_city_12_a', 'zurich_city_13_a', 'zurich_city_13_b', 'zurich_city_14_a', 'zurich_city_14_b', 
                          'zurich_city_14_c', 'zurich_city_15_a']

    main()