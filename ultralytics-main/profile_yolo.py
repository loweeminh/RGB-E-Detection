import torch
import numpy as np
from calflops import calculate_flops
from ultralytics import YOLO

def run_benchmark(model_path, input_shapes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = YOLO(model_path).model.to(device).eval()
    
    inputs = [torch.randn(shape).to(device) for shape in input_shapes]
    
    flops, macs, params = calculate_flops(
        model=model, args=inputs, output_as_string=True,
        print_detailed=False, output_precision=4
    )

    avg_ms, fps = 0.0, 0.0
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = []

        with torch.inference_mode():
            # Warm-up to stabilize GPU clocks
            for _ in range(50): model(*inputs)
            
            for _ in range(repetitions):
                starter.record()
                model(*inputs)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
        
        avg_ms = np.mean(timings)
        fps = 1000 / avg_ms

    print(f"\n{' BENCHMARK REPORT ':=^50}")
    print(f"Device  : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"Params  : {params}")
    print(f"FLOPs   : {flops}")
    print(f"Latency : {avg_ms:.3f} ms")
    print(f"FPS     : {fps:.2f}")
    print(f"{'':=^50}\n")

if __name__ == "__main__":
    RUN_CONFIG = {
        "model_path": "/home/loweeminh/RGB-E-Detection/ultralytics-main/yolo11l_fusion.yaml",
        "input_shapes": [(1, 3, 480, 640), (1, 3, 480, 640)]
    }
    run_benchmark(**RUN_CONFIG)