# Jetson Nano Inference Toolkit

This repository provides scripts for running **object detection** inference using ONNX models on the NVIDIA Jetson Nano. The code is optimized for Jetson's hardware constraints and supports both live streams and batch processing.

## Features

- **Object Detection**: Inference on images, videos, live streams, and folders of images.

---

## Requirements

- **Hardware**: NVIDIA Jetson Nano (or similar Jetson device)
- **Software**: Jetpack 4.6.6
- **OS**: Ubuntu (JetPack recommended)
- **Python**: 3.6 (required for the provided ONNX Runtime wheel)
- **ONNX Models**: You must provide your own ONNX models for object detection.

### ONNX Runtime Installation for Jetson Nano

To install ONNX Runtime GPU version 1.11.0 for Jetson Nano, run the following commands:

```bash
wget https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.whl -O onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
pip3 install onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
```
After installation, you should see:

```
Successfully installed onnxruntime-gpu-1.11.0
```

### Python Dependencies

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**Note:**  
- The `onnxruntime-gpu` package is installed from a local wheel file. You must have the file `onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl` at the specified path or update the path in `requirements.txt` to match your environment.
- If you are not on Jetson Nano, you may need a different ONNX Runtime wheel.

---

## Usage

### 1. Object Detection

Run the main script:

```bash
python objectdetection.py
```

You will be prompted to choose a mode:

- `image`: Run detection on a single image.
- `video`: Run detection on a video file.
- `live`: Run detection on a webcam.
- `folder`: Batch process all images in a folder.

**You will need to provide:**
- The path to your ONNX detection model.
- The class names for your model (edit the `class_names` list in the script).
- Input/output file or folder paths as prompted.


**Templete**
 - This is the command-line templete
 ```bash
   python3 objectdetection.py --mode ['image', 'folder', 'video', 'live'] --source ['0','1','2',...] --model-path [model path to your onnx model file] --classes [your classes with a space] --batch-size [1,2,3,4] --input-size [widthxheight] --score-threshold [0-1]
 ```
 - This is the example code
 ```bash
    python3 objectdetection.py --mode live --source 0 --model-path /home/jetson/Jetson_codes/models/rtmtinyface.onnx --classes faces --batch-size 1 --input-size 640 640 --score-threshold 0.6
 ``` 

## Model Preparation

- This toolkit does **not** include any ONNX models. You must export or download your own models compatible with ONNX Runtime.
- For best results, use models with input shapes and normalization compatible with the preprocessing in the scripts.
- Example model path for objectdetecion (edit as needed in `objectdection.py`):
  ```
  model_path = '/home/jetson/Downloads/InferenceCodes/models/rtmtinyface.onnx'
  ```

---

## Notes

- **Performance**: The scripts are optimized for Jetson Nano, including frame skipping and memory monitoring.
- **Class Names**: Update the `class_names` list in each script to match your model's classes.
- **Output**: Results are saved in the current directory or specified output folders. For segmentation, overlays and masks are saved separately.
- **Interactive Mode**: In object detection, you can step through images and save results manually or process all automatically.

---
