import sys
import os

sys.path.append(os.path.abspath("ultralytics-main"))
from ultralytics import YOLO

# Paper "Adaptive Token Fusion Transformer for Multi-Modal Object Detection with RGB and Event Data"
def main(mode):
    if mode == "event":
        # event data single modality inference
        model = YOLO("runs/detect/event/weights/best.pt")
        for route in g_folder_names_train:
            img_path = "datasets/dsec_events/train/images/" + route
            results = model(img_path, save_txt=True, stream=True, sensor="event")

            for result in results:
                result.save()

        for route in g_folder_names_val:
            img_path = "datasets/dsec_events/val/images/" + route
            results = model(img_path, save_txt=True, stream=True, sensor="event")

            for result in results:
                result.save()
    elif mode == "rgb":
        # rgb data single modality inference
        model = YOLO("runs/detect/rgb/weights/best.pt")
        for route in g_folder_names_train:
            img_path = "datasets/dsec_rgb/train/images/" + route
            results = model(img_path, save_txt=True, stream=True, sensor="rgb")

            for result in results:
                result.save()

        for route in g_folder_names_val:
            img_path = "datasets/dsec_rgb/val/images/" + route
            results = model(img_path, save_txt=True, stream=True, sensor="rgb")

            for result in results:
                result.save()
    elif mode == "fusion":
        # fusion model inference
        model = YOLO("runs/detect/ATFT/weights/best.pt")
        for route in g_folder_names_train:
            img_path = "datasets/dsec_events/train/images/" + route
            img_path2 = "datasets/dsec_rgb/train/images/" + route
            results = model(img_path, save_txt=True, stream=True, sensor="fusion", source2=img_path2)

            for result in results:
                result.save()

        for route in g_folder_names_val:
            img_path = "datasets/dsec_events/val/images/" + route
            img_path2 = "datasets/dsec_rgb/val/images/" + route
            results = model(img_path, save_txt=True, stream=True, sensor="fusion", source2=img_path2)

            for result in results:
                result.save()

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
    
    main(mode="fusion")