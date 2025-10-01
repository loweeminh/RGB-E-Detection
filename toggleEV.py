import numpy as np
import tables
import os
from tqdm import tqdm

# Function to apply min max normalization to the array
def Normalize_MinMax(data, global_max=255.0):
    data = np.clip(data, 0, global_max)
    nor_data = (data / global_max) * 255.0
    return nor_data.astype(np.float16)

# Function to remove all files in the output directory
def Del_Old_Files(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Function to save the event stack as a .npy file
def Save_Event_Stacks(event_stack_3d, count, output_dir):
    save_path = os.path.join(output_dir, f'{count:06}.npy')
    np.save(save_path, event_stack_3d)

# Function to process event data, generate event stacks, and save them
def Process_Event_Data(events_file, timestamps_file, output_dir, frame_size, chunk_size):
    Del_Old_Files(output_dir)
    
    with tables.open_file(events_file, mode='r') as file:
        total_events = file.get_node('/events/x').shape[0]
        t_o = file.get_node('/t_offset').read()
        timestamps = np.loadtxt(timestamps_file)

        pos_count = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint16)
        neg_count = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint16)
        toggle_count = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint16)
        last_polarity = np.full((frame_size[1], frame_size[0]), -1, dtype=np.int16)

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
                        curr_polarity = p[n]
                        if p[n] == 1:
                            pos_count[y[n], x[n]] += 1
                        else:
                            neg_count[y[n], x[n]] += 1
                        if last_polarity[y[n], x[n]] != -1 and last_polarity[y[n], x[n]] != curr_polarity:
                            toggle_count[y[n], x[n]] += 1
                        last_polarity[y[n], x[n]] = curr_polarity
                    else:
                        # Skip the first and last event stacks
                        if i != 0 and i != (len(timestamps) - 1):
                            nor_pos_count = Normalize_MinMax(pos_count, 1209)
                            nor_neg_count = Normalize_MinMax(neg_count, 461)
                            nor_toggle_count = Normalize_MinMax(toggle_count, 527)
                            event_stack_3d = np.stack((nor_pos_count, nor_neg_count, nor_toggle_count), axis=-1)
                            Save_Event_Stacks(event_stack_3d, i - 1, output_dir)
                        pos_count.fill(0)
                        neg_count.fill(0)
                        toggle_count.fill(0)
                        prev = n
                        break_flg = True
                        break
                if not break_flg:
                    last = i
                    break

# Main function
def main():
    # Process event data, generate event stacks, and save them
    for route in tqdm(g_folder_names_train, desc="Train Routes"):
        events_file = 'DSEC-Det/train/' + route + '/events/events.h5'
        timestamps_file = 'DSEC-Det/label/train/' + route + '/timestamps.txt'
        output_dir = 'datasets/dsec_events/train/images/' + route
        Process_Event_Data(events_file, timestamps_file, output_dir, g_frame_size, g_chunk_size)
    
    for route in tqdm(g_folder_names_val, desc="Val Routes"):
        events_file = 'DSEC-Det/val/' + route + '/events/events.h5'
        timestamps_file = 'DSEC-Det/label/val/' + route + '/timestamps.txt'
        output_dir = 'datasets/dsec_events/val/images/' + route
        Process_Event_Data(events_file, timestamps_file, output_dir, g_frame_size, g_chunk_size)

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

    main()