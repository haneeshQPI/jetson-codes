# Jetson Nano Inference Toolkit

This repository provides scripts for running **object detection** and **semantic segmentation** inference using ONNX models on the NVIDIA Jetson Nano. The code is optimized for Jetson's hardware constraints and supports both live streams and batch processing.

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
- `live`: Run detection on a webcam or network stream (RTSP/HTTP).
- `folder`: Batch process all images in a folder.
- `interactive`: Step through images in a folder with preview and manual save.

**You will need to provide:**
- The path to your ONNX detection model.
- The class names for your model (edit the `class_names` list in the script).
- Input/output file or folder paths as prompted.

### 2. Semantic Segmentation

Run the segmentation script:

```bash
python segmentation.py
```

You will be prompted to choose a mode:

- `image`: Run segmentation on a single image.
- `video`: Run segmentation on a video file.

**You will need to provide:**
- The path to your ONNX segmentation model (edit `model_path` in the script).
- The class names for your model (edit the `class_names` list in the script).
- Input/output file paths as prompted.

---

## Model Preparation

- This toolkit does **not** include any ONNX models. You must export or download your own models compatible with ONNX Runtime.
- For best results, use models with input shapes and normalization compatible with the preprocessing in the scripts.
- Example model path for segmentation (edit as needed in `segmentation.py`):
  ```
  model_path = '/home/jetson/Downloads/InferenceCodes/models/deeplabkitty.onnx'
  ```

---

## Notes

- **Performance**: The scripts are optimized for Jetson Nano, including frame skipping and memory monitoring.
- **Class Names**: Update the `class_names` list in each script to match your model's classes.
- **Output**: Results are saved in the current directory or specified output folders. For segmentation, overlays and masks are saved separately.
- **Interactive Mode**: In object detection, you can step through images and save results manually or process all automatically.

---

## Troubleshooting

- If you encounter errors loading the ONNX model, ensure the model is compatible with ONNX Runtime and Jetson Nano.
- If you see errors about missing dependencies, double-check your Python version and the installed packages.
- For issues with the ONNX Runtime wheel, download the correct version for your Jetson Nano and Python version from the [official ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases).

---

## License

This project is provided as-is for research and educational purposes.

---

**Feel free to modify the scripts and README to fit your specific models and use cases!** 