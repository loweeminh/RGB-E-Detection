import numpy as np
import tables
import os
import math
from tqdm import tqdm

# Function to remove all files in the output directory
def Del_Old_Files(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Function to save the event stack as a numpy file
def Save_Event_Stacks(event_stack_3d, count, output_dir):
    save_path = os.path.join(output_dir, f'{count:06}.npy')
    np.save(save_path, event_stack_3d)

# Function to process event data, generate event stacks, and save them
def Process_Event_Data(events_file, timestamps_file, output_dir, frame_size, chunk_size, bin_num):
    init_flg = True
    Del_Old_Files(output_dir)
    
    with tables.open_file(events_file, mode='r') as file:
        total_events = file.get_node('/events/x').shape[0]
        t_o = file.get_node('/t_offset').read()
        timestamps = np.loadtxt(timestamps_file)

        voxel_grid = np.full((frame_size[1], frame_size[0], bin_num * 2), 0, dtype=np.uint16)

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
                t_max = timestamps[i]
                break_flg = False 
                for n in range(prev, len(t)):
                    if init_flg:
                        t_min = t[n]
                        init_flg = False
                    if t[n] <= timestamps[i]:
                        bin_order = math.floor(((t[n] - t_min) / (t_max - t_min)) * bin_num)
                        if bin_order == bin_num:
                            bin_order = bin_num - 1
                        if p[n] == 1:
                            ch = bin_order * 2
                        else:
                            ch = bin_order * 2 + 1
                        voxel_grid[y[n], x[n], ch] += 1
                    else:
                        if i != 0 and i != (len(timestamps) - 1):
                            clipped_voxel_grid = np.clip(voxel_grid, 0, 255).astype(np.uint8)
                            Save_Event_Stacks(clipped_voxel_grid, i - 1, output_dir)
                        voxel_grid.fill(0)
                        t_min = timestamps[i]
                        prev = n
                        break_flg = True
                        break
                if not break_flg:
                    last = i
                    break

# Main function
def main():
    # Process training data
    for route in tqdm(g_folder_names_train, desc="Train Routes"):
        events_file = '../DSEC-Det/train/' + route + '/events/events.h5'
        timestamps_file = '../DSEC-Det/label/train/' + route + '/timestamps.txt'
        output_dir = '../datasets/dsec_events/train/images/' + route
        Process_Event_Data(events_file, timestamps_file, output_dir, g_frame_size, g_chunk_size, g_bin_num)
    
    # Process validation data
    for route in tqdm(g_folder_names_val, desc="Val Routes"):
        events_file = '../DSEC-Det/val/' + route + '/events/events.h5'
        timestamps_file = '../DSEC-Det/label/val/' + route + '/timestamps.txt'
        output_dir = '../datasets/dsec_events/val/images/' + route
        Process_Event_Data(events_file, timestamps_file, output_dir, g_frame_size, g_chunk_size, g_bin_num)

if __name__ == "__main__":
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
    
    g_frame_size = (640, 480) # (width, height)
    g_chunk_size = 10000 # split loaded data due to limited RAM
    g_bin_num = 6

    main()