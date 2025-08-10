# Predicting Plant Growth Structures with Multimodal Data Using Convolutional Long Short-Term Memory (ConvLSTM)

A multimodal ConvLSTM model that predicts the next plant-growth frame from image sequences and environmental parameters, using a unified MSE + Temporal Consistency (TC) loss to balance visual fidelity and smooth temporal dynamics.  
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## Overview
This project introduces a multimodal deep learning architecture that uses ConvLSTM networks to predict plant growth structures from past appearances and environmental factors (e.g., temperature, soil moisture, luminosity, soil pH). It also introduces an improved loss function that unifies Mean Squared Error (MSE) and Temporal Consistency (TC) to capture pixel-level detail and frame-to-frame smoothness during training. The long-term vision is to support agricultural digital twins so farmers can trial changes virtually before applying them in the field.

## Features
- Multimodal inputs: image sequences plus aligned environmental parameters  
- ConvLSTM backbone for spatiotemporal modeling of plant morphology  
- Unified loss (MSE + TC): α = 1.0, β = 0.2 (code-adjustable)  
- Metrics: MSE, PSNR, SSIM, TC, and Total Variation (TV)

## Data
- Composition: 64 sequences of lettuce; images captured at 12-hour intervals  
- Preprocessing: YOLOv8 instance segmentation to isolate plant pixels; images resized to (H, W) = (95, 126); parameters standardized (global mean/variance)  
- Sequence length: fixed L = 45 via pre-padding when sequences are shorter

## Model Architecture
- Image stream: two ConvLSTM2D layers with BatchNorm over past frames  
- Parameter stream: TimeDistributed dense embedding of the 4-D parameter vector, tiled to spatial dimensions  
- Fusion: concatenate (image features ⊕ tiled parameter embedding) → Conv3D → next-frame prediction (sigmoid)

## Unified Loss
$$
L_{\text{unified}} = \alpha\,L_{\text{MSE}} + \beta\,L_{\text{TC}},\quad \alpha=1.0,\ \beta=0.2
$$
- MSE enforces pixel fidelity.  
- TC penalizes abrupt temporal changes to promote smoothness.

## Results (test set)
The unified loss achieves strong visual quality while keeping temporal artifacts low. (Representative results:)

| Model          | MSE     | PSNR    | SSIM   | TC      | TV       |
|----------------|---------|---------|--------|---------|----------|
| MSE-only       | 0.0001  | 45.1848 | 0.9633 | 0.0001  | 148.1984 |
| TC-only        | 0.0326  | 14.8740 | 0.0162 | 0.0000  | 239.4765 |
| Unified (ours) | 0.0001  | 46.9794 | 0.9692 | 0.0001  | 141.8375 |

## Installation
Easiest path is Google Colab (this script was authored from a Colab workflow). For local runs, install requirements and set paths.

```bash
# (Optional) create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
Edit these variables near the top of `main.py` to point to your data and YOLOv8 weights:
```python
dataset_dir = "/path/to/dataset_root"
segmentation_model = YOLO("/path/to/yolov8_plant_segmentation_best.pt")
trained_model_dir = "./models"  # where to save trained weights and artifacts
```

Then run training/evaluation (Colab cell or local Python):
```bash
python main.py
```

Default training configuration (from code):
- MAX_SEQUENCE_LENGTH = 45, BATCH_SIZE = 2  
- IMAGE_WIDTH = 126, IMAGE_HEIGHT = 95, IMAGE_CHANNELS = 3  
- LEARNING_RATE = 0.001, epochs = 100  
- Windowed prediction example in code: window_size = 30, num_predictions = 10  
- Three training runs: MSE-only, TC-only, and Unified (MSE+TC)

The file includes helper/experimental functions; only the invoked training/evaluation paths are documented here.

## License
This repository is licensed under the MIT License. See `LICENSE`.