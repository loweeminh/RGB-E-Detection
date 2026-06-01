import numpy as np
import tables
import os
from tqdm import tqdm

def normalize_min_max(data, global_max=255.0):
    """Apply min-max normalization and convert to float16."""
    data = np.clip(data, 0, global_max)
    return ((data / global_max) * 255.0).astype(np.float16)

def clear_directory(directory):
    """Create directory if not exists or remove all existing files inside."""
    os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def save_event_stack(stack, index, output_dir):
    """Save the 3D event stack as a .npy file."""
    save_path = os.path.join(output_dir, f'{index:06}.npy')
    np.save(save_path, stack)

def process_event_data(events_file, timestamps_file, output_dir, frame_size, chunk_size):
    """Main logic for processing H5 event data into normalized event stacks."""
    clear_directory(output_dir)
    
    with tables.open_file(events_file, mode='r') as file:
        total_events = file.get_node('/events/x').shape[0]
        t_offset = file.get_node('/t_offset').read()
        timestamps = np.loadtxt(timestamps_file)

        # Initialize accumulators
        pos_count = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint16)
        neg_count = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint16)
        toggle_count = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint16)
        last_p = np.full((frame_size[1], frame_size[0]), -1, dtype=np.int16)

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
                            pos_count[y[n], x[n]] += 1
                        else:
                            neg_count[y[n], x[n]] += 1
                        
                        if last_p[y[n], x[n]] != -1 and last_p[y[n], x[n]] != curr_p:
                            toggle_count[y[n], x[n]] += 1
                        last_p[y[n], x[n]] = curr_p
                    else:
                        # Save stack if not at the very beginning/end
                        if 0 < i < (len(timestamps) - 1):
                            stack = np.stack((
                                normalize_min_max(pos_count, 1209),
                                normalize_min_max(neg_count, 461),
                                normalize_min_max(toggle_count, 527)
                            ), axis=-1)
                            save_event_stack(stack, i - 1, output_dir)
                        
                        pos_count.fill(0)
                        neg_count.fill(0)
                        toggle_count.fill(0)
                        chunk_ptr = n
                        finished_stack = True
                        break
                
                if not finished_stack:
                    last_ts_idx = i
                    break

def main():
    """Iterate through data splits and automatically detect routes to process."""
    base_in = '/home/loweeminh/Downloads/DSEC-Det'
    base_out = 'dsec_datasets/events'
    frame_size = (640, 480)
    chunk_size = 10000

    for split in ['train', 'val']:
        input_path = os.path.join(base_in, split)
        if not os.path.exists(input_path):
            continue
            
        # Automatically list all subdirectories in train/val
        routes = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
        
        for route in tqdm(routes, desc=f"Processing {split}"):
            events_h5 = os.path.join(base_in, split, route, 'events/events.h5')
            ts_txt = os.path.join(base_in, 'label', split, route, 'timestamps.txt')
            out_path = os.path.join(base_out, split, 'images', route)
            
            if os.path.exists(events_h5) and os.path.exists(ts_txt):
                process_event_data(events_h5, ts_txt, out_path, frame_size, chunk_size)

if __name__ == "__main__":
    main()