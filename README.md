<h1 align="center">
Adaptive Token Fusion Transformer for Multi-Modal Object Detection with RGB and Event Data
</h1>

Official code repository for my research paper "Adaptive Token Fusion Transformer for Multi-Modal Object Detection with RGB and Event Data" at National Taiwan University of Science and Technology, HIS-LAB.

![Alt text](images/model_framework.png)

## Abstract

Object detection under challenging conditions,such as low illumination and high dynamic range, remains alimitation for conventional frame-based cameras. Eventcameras, which asynchronously record pixel-level brightnesschanges with high temporal resolution and wide dynamic range,offer a complementary sensing modality. In this work, weintroduce a novel framework that encourages joint learning ofRGB and event data within a shared feature space using theTransformer. At the preprocessing step, ToggleEV eventrepresentation combines positive and negative event counts witha polarity transition map, enabling precise encoding of spatialevent distributions and motion patterns in a compact structuredformat. The Adaptive Token Fusion Transformer (ATFT)modules operate across multiple feature scales, leveragingattention mechanisms to integrate informative tokens andsuppress cross-modal noise. Extensive experiments on the publicDSEC-Detection dataset demonstrate that our fusion methodoutperforms the state-of-the-art approach.

## Package Installation

To install the required packages, run:

```
pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Dataset

DSEC-Detection dataset extends the original DSEC collection by providing high-quality bounding box annotations on sequences recorded using a multi-sensor driving platform equipped with Prophesee Gen3.1 event cameras and synchronized FLIR Blackfly S color cameras. It comprises 60 sequences in total, divided into 47 training sequences (63,239 frames) and 13 test sequences (15,105 frames).

DSEC-Detection is available here: [https://dsec.ifi.uzh.ch](https://dsec.ifi.uzh.ch)

Organize them into the following directory structure:

```
└── DSEC-Det
    ├── label
    |   ├── train
    |   |   ├── interlaken__00_c
    |   |   |   ├── timestamps.txt
    |   |   |   └── tracks.npy
    |   |   └── ...
    |   └── val
    |       ├── interlaken__00_a
    |       |   ├── timestamps.txt
    |       |   └── tracks.npy
    |       └── ...
    ├── train
    |   ├── interlaken__00_c
    |   |   ├── events
    |   |   |   └── events.h5
    |   |   └── rgb
    |   |       ├── 000000.png
    |   |       └── ...
    |   └── ...
    └── val
        ├── interlaken__00_a
        |   ├── events
        |   |   └── events.h5
        |   └── rgb
        |       ├── 000000.png
        |       └── ...
        └── ...
```

Implement ToggleEV event representation to preprocess raw event data:
```
python3 toggleEV.py
```

Generate labels and clean image files for downstream processes:
```
python3 toolbox.py
```

## Training

To train ATFT model, run:
```
python3 train_fusion.py
```

## Validation

To validate ATFT model using our pretrained weights, run:
```
python3 val_fusion.py
```

## Code Acknowledgments
This project has used code from the following project:
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)