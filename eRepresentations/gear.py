import numpy as np
import tables
import os
from tqdm import tqdm
from PIL import Image

# Function to remove all files in the output directory
def Del_Old_Files(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Function to save the event stack as a .npy file
def Save_Event_Stacks(event_stack_3d, count, output_dir):
    save_path = os.path.join(output_dir, f'{count:06}.png')
    # Convert float16 → uint8
    event_stack_uint8 = np.clip(event_stack_3d, 0, 255).astype(np.uint8)
    # Save as PNG (RGB)
    img = Image.fromarray(event_stack_uint8, mode='RGB')
    img.save(save_path, format='PNG')

# Function to process event data, generate event stacks, and save them
def Process_Event_Data(events_file, timestamps_file, output_dir, frame_size, chunk_size):
    Del_Old_Files(output_dir)
    
    with tables.open_file(events_file, mode='r') as file:
        total_events = file.get_node('/events/x').shape[0]
        t_o = file.get_node('/t_offset').read()
        timestamps = np.loadtxt(timestamps_file)

        event_total = np.full((frame_size[1], frame_size[0]), 0, dtype=np.uint32)
        event_current = np.full((frame_size[1], frame_size[0]), 0, dtype=np.float32)
        event_previous = np.full((frame_size[1], frame_size[0]), 0, dtype=np.float32)
        event_accumulation = np.full((frame_size[1], frame_size[0]), 128, dtype=np.int32)

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
                        event_accumulation[y[n], x[n]] += ((2 * int(p[n]) - 1) * 8)
                        event_total[y[n], x[n]] += 1
                    else:
                        # Skip the first and last event stacks
                        if i != 0 and i != (len(timestamps) - 1):
                            f_t = np.exp((-1.0) * event_total / 10)
                            event_current = (f_t * event_previous) + event_accumulation
                            event_stack_3d = np.stack(((np.tanh(event_current / 128.0) * 128 + 128), (np.tanh(event_previous / 128.0) * 128 + 128), (np.tanh(event_total / 32) * 255)), axis=-1)
                            event_previous = event_current
                            Save_Event_Stacks(event_stack_3d, i - 1, output_dir)
                        event_total.fill(0)
                        event_accumulation.fill(0)
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
        events_file = '../DSEC-Det/train/' + route + '/events/events.h5'
        timestamps_file = '../DSEC-Det/label/train/' + route + '/timestamps.txt'
        output_dir = '../datasets/dsec_events/train/images/' + route
        Process_Event_Data(events_file, timestamps_file, output_dir, g_frame_size, g_chunk_size)
    
    for route in tqdm(g_folder_names_val, desc="Val Routes"):
        events_file = '../DSEC-Det/val/' + route + '/events/events.h5'
        timestamps_file = '../DSEC-Det/label/val/' + route + '/timestamps.txt'
        output_dir = '../datasets/dsec_events/val/images/' + route
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