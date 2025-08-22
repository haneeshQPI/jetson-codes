import onnxruntime as ort
import numpy as np
import cv2
import time
import argparse
from pathlib import Path
import glob
import os
import json
import yaml

def setup_jetson_session(model_path):
    """Setup optimized ONNX session for Jetson Nano"""
    providers = []
    
    try:
        providers.append(('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'HEURISTIC',
        }))
    except:
        pass
    
    providers.append('CPUExecutionProvider')
    
    session = ort.InferenceSession(model_path, providers=providers)
    return session

def load_config_file(config_path):
    """Load configuration from JSON or YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return config
    except Exception as e:
        raise ValueError(f"Error loading config file {config_path}: {e}")

def find_model_files(search_paths=None):
    """Find available ONNX model files"""
    if search_paths is None:
        search_paths = [
            './models',
            '/opt/models'
        ]
    
    model_files = []
    for search_path in search_paths:
        path = Path(search_path)
        if path.exists():
            model_files.extend(path.glob('*.onnx'))
    
    return sorted(set(model_files))

def load_classes_file(classes_path):
    """Load class names from a text file"""
    classes_path = Path(classes_path)
    
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_path}")
    
    try:
        with open(classes_path, 'r') as f:
            # Support different formats
            content = f.read().strip()
            
            # Try JSON format first
            try:
                classes = json.loads(content)
                if isinstance(classes, list):
                    return classes
                elif isinstance(classes, dict) and 'classes' in classes:
                    return classes['classes']
            except json.JSONDecodeError:
                pass
            
            # Try YAML format
            try:
                classes = yaml.safe_load(content)
                if isinstance(classes, list):
                    return classes
                elif isinstance(classes, dict) and 'classes' in classes:
                    return classes['classes']
            except yaml.YAMLError:
                pass
            
            # Fallback to simple text format (one class per line)
            lines = content.split('\n')
            classes = [line.strip() for line in lines if line.strip()]
            return classes
            
    except Exception as e:
        raise ValueError(f"Error loading classes file {classes_path}: {e}")

def detect_model_config(model_path):
    """Try to detect configuration files for a model"""
    model_path = Path(model_path)
    model_dir = model_path.parent
    model_name = model_path.stem
    
    # Look for config files
    possible_configs = [
        model_dir / f"{model_name}.json",
        model_dir / f"{model_name}.yaml",
        model_dir / f"{model_name}.yml",
        model_dir / "config.json",
        model_dir / "config.yaml",
        model_dir / "config.yml"
    ]
    
    # Look for classes files
    possible_classes = [
        model_dir / f"{model_name}_classes.txt",
        model_dir / f"{model_name}.names",
        model_dir / "classes.txt",
        model_dir / "classes.names",
        model_dir / "labels.txt"
    ]
    
    config_file = None
    classes_file = None
    
    for config in possible_configs:
        if config.exists():
            config_file = config
            break
    
    for classes in possible_classes:
        if classes.exists():
            classes_file = classes
            break
    
    return config_file, classes_file

def get_default_classes():
    """Get default class names for common detection models"""
    return {
        'coco': [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ],
        'vehicle': ['bus', 'car', 'bike', 'road', 'person'],
        'face': ['face'],
        'person': ['person']
    }
    """Setup optimized ONNX session for Jetson Nano"""
    providers = []
    
    try:
        providers.append(('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'HEURISTIC',
        }))
    except:
        pass
    
    providers.append('CPUExecutionProvider')
    
    session = ort.InferenceSession(model_path, providers=providers)
    return session

class SequentialBatchProcessor:
    """Sequential batch processor - no threading"""
    
    def __init__(self, session, batch_size, target_size, class_names, score_threshold):
        self.session = session
        self.batch_size = batch_size
        self.target_size = target_size
        self.class_names = class_names
        self.score_threshold = score_threshold
        
    def process_single_image(self, image_path, output_path=None, save_image=True):
        """Process a single image"""
        print(f"Processing single image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        start_time = time.time()
        
        # Preprocess
        processed, info = self.preprocess_frame(image)
        
        # Run inference
        batch_tensor = np.expand_dims(processed, axis=0)  # Add batch dimension
        input_name = self.session.get_inputs()[0].name
        inference_start = time.time()
        results = self.session.run(None, {input_name: batch_tensor})
        inference_time = time.time() - inference_start
        
        # Postprocess
        batch_results = self.postprocess_batch(results, [info])
        boxes, scores, class_ids = batch_results[0]
        
        # Draw detections
        output_image = self.draw_detections(image.copy(), boxes, scores, class_ids)
        
        total_time = time.time() - start_time
        
        # Print results
        print(f"Inference time: {inference_time*1000:.1f}ms")
        print(f"Total time: {total_time*1000:.1f}ms")
        print(f"Detections found: {len(boxes)}")
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            print(f"  {i+1}. {class_name}: {score:.3f} at {box}")
        
        # Save output image
        if save_image:
            if output_path is None:
                # Generate output path
                input_path = Path(image_path)
                output_path = input_path.parent / f"{input_path.stem}_detected{input_path.suffix}"
            
            cv2.imwrite(str(output_path), output_image)
            print(f"Output saved to: {output_path}")
        
        return output_image, boxes, scores, class_ids
    
    def process_image_folder(self, folder_path, output_folder=None, image_extensions=None):
        """Process all images in a folder"""
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        print(f"Processing image folder: {folder_path}")
        
        # Find all image files
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Input folder does not exist: {folder_path}")
        
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
            image_files.extend(folder_path.glob(f"*{ext.upper()}"))
        
        image_files.sort()
        
        if not image_files:
            print(f"No image files found in the folder with extensions: {image_extensions}")
            return []
        
        print(f"Found {len(image_files)} images")
        
        # Setup output folder
        if output_folder is None:
            output_folder = folder_path / "detected_output"
        else:
            output_folder = Path(output_folder)
        
        print(f"Creating output folder: {output_folder}")
        output_folder.mkdir(exist_ok=True)
        
        # Process images in batches
        results = []
        total_start_time = time.time()
        total_detections = 0
        
        for i in range(0, len(image_files), self.batch_size):
            batch_files = image_files[i:i + self.batch_size]
            batch_images = []
            batch_info = []
            valid_files = []
            
            # Load and preprocess batch
            for file_path in batch_files:
                try:
                    image = cv2.imread(str(file_path))
                    if image is not None:
                        processed, info = self.preprocess_frame(image)
                        batch_images.append(processed)
                        batch_info.append(info)
                        valid_files.append((file_path, image))
                    else:
                        print(f"Warning: Could not load {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            
            if not batch_images:
                continue
            
            # Run batch inference
            batch_tensor = np.stack(batch_images, axis=0)
            batch_start = time.time()
            input_name = self.session.get_inputs()[0].name
            batch_results = self.session.run(None, {input_name: batch_tensor})
            batch_time = time.time() - batch_start
            
            # Postprocess batch
            detections = self.postprocess_batch(batch_results, batch_info)
            
            # Save results
            for (file_path, original_image), (boxes, scores, class_ids) in zip(valid_files, detections):
                # Draw detections
                output_image = self.draw_detections(original_image.copy(), boxes, scores, class_ids)
                
                # Save output
                output_filename = f"{file_path.stem}_detected{file_path.suffix}"
                output_path = output_folder / output_filename
                
                try:
                    cv2.imwrite(str(output_path), output_image)
                    print(f"Processed: {file_path.name} -> {len(boxes)} detections -> {output_filename}")
                except Exception as e:
                    print(f"Error saving {output_path}: {e}")
                    continue
            
            # Progress update
            processed_count = min(i + self.batch_size, len(image_files))
            progress = (processed_count / len(image_files)) * 100
            print(f"Batch {i//self.batch_size + 1}: {len(batch_images)} images in {batch_time*1000:.1f}ms | "
                  f"Progress: {progress:.1f}%")
        
        total_time = time.time() - total_start_time
        total_detections = sum(r['detections'] for r in results)
        
        print(f"\nFolder processing complete!")
        print(f"Total images processed: {len(results)}")
        print(f"Total detections: {total_detections}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per image: {total_time/len(results):.2f}s")
        print(f"Output folder: {output_folder}")
        
        return results
    
    def process_video_sequential(self, video_path, output_path):
        """Process video sequentially in batches - no threading"""
        print(f"Sequential batch processing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_batch = []
        frame_count = 0
        start_time = time.time()
        
        print(f"Processing {total_frames} frames in batches of {self.batch_size}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_batch.append(frame.copy())
            
            # Process when batch is full or at end of video
            if len(frame_batch) == self.batch_size or frame_count == total_frames - 1:
                
                # Preprocess batch
                batch_input = []
                batch_info = []
                
                for f in frame_batch:
                    processed, info = self.preprocess_frame(f)
                    batch_input.append(processed)
                    batch_info.append(info)
                
                # Stack into batch tensor
                if len(batch_input) > 0:
                    batch_tensor = np.stack(batch_input, axis=0)
                    
                    # Single inference call for entire batch
                    batch_start = time.time()
                    input_name = self.session.get_inputs()[0].name
                    results = self.session.run(None, {input_name: batch_tensor})
                    batch_time = time.time() - batch_start
                    
                    # Postprocess each frame in batch
                    batch_results = self.postprocess_batch(results, batch_info)
                    
                    # Draw and write results
                    for original_frame, (boxes, scores, class_ids) in zip(frame_batch, batch_results):
                        output_frame = self.draw_detections(original_frame, boxes, scores, class_ids)
                        out.write(output_frame)
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    progress = (frame_count / total_frames) * 100
                    
                    print(f"Batch processed: {len(frame_batch)} frames in {batch_time*1000:.1f}ms | "
                          f"Progress: {progress:.1f}% | FPS: {fps_current:.1f}")
                
                # Clear batch
                frame_batch.clear()
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        total_time = time.time() - start_time
        print(f"Sequential processing complete: {frame_count} frames in {total_time:.1f}s")
        print(f"Average FPS: {frame_count / total_time:.1f}")
    
    def process_live_sequential(self, source):
        """Process live stream sequentially - accumulate then process"""
        print(f"Sequential live processing from: {source}")
        
        cap = cv2.VideoCapture(source)
        frame_batch = []
        frame_count = 0
        
        print(f"Collecting {self.batch_size} frames, then processing...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_batch.append(frame.copy())
            frame_count += 1
            
            # Show current frame while collecting
            cv2.putText(frame, f"Collecting frames: {len(frame_batch)}/{self.batch_size}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Sequential Processing', frame)
            
            # Process batch when full
            if len(frame_batch) == self.batch_size:
                print(f"Processing batch of {len(frame_batch)} frames...")
                
                # Preprocess batch
                batch_input = []
                batch_info = []
                
                for f in frame_batch:
                    processed, info = self.preprocess_frame(f)
                    batch_input.append(processed)
                    batch_info.append(info)
                
                # Run batch inference
                batch_tensor = np.stack(batch_input, axis=0)
                batch_start = time.time()
                input_name = self.session.get_inputs()[0].name
                results = self.session.run(None, {input_name: batch_tensor})
                batch_time = time.time() - batch_start
                
                # Postprocess and display results
                batch_results = self.postprocess_batch(results, batch_info)
                
                print(f"Batch inference: {batch_time*1000:.1f}ms")
                
                # Show all processed frames quickly
                for i, (original_frame, (boxes, scores, class_ids)) in enumerate(zip(frame_batch, batch_results)):
                    output_frame = self.draw_detections(original_frame, boxes, scores, class_ids)
                    
                    # Add batch info
                    cv2.putText(output_frame, f"Batch result {i+1}/{len(frame_batch)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(output_frame, f"Detections: {len(boxes)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    cv2.imshow('Sequential Processing', output_frame)
                    cv2.waitKey(200)  # Show each result for 200ms
                
                # Clear batch
                frame_batch.clear()
                print("Ready for next batch...")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def preprocess_frame(self, frame):
        """Preprocess single frame"""
        target_w, target_h = self.target_size
        
        # Convert and resize
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        scale = min(target_h / h, target_w / w)
        
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image_rgb, (new_w, new_h))
        
        # Pad
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        padded[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        # Normalize and transpose
        image_float = padded.astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        image_normalized = (image_float - mean) / std
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        
        preprocess_info = {
            'scale': scale,
            'offset_x': start_x,
            'offset_y': start_y,
            'original_size': (w, h)
        }
        
        return image_transposed, preprocess_info
    
    def postprocess_batch(self, results, batch_info):
        """Postprocess batch results"""
        if len(results) != 2:
            return [[] for _ in batch_info]
        
        batch_dets, batch_labels = results
        batch_results = []
        
        for i, info in enumerate(batch_info):
            dets = batch_dets[i] if len(batch_dets.shape) > 2 else batch_dets
            labels = batch_labels[i] if len(batch_labels.shape) > 1 else batch_labels
            
            boxes, scores, class_ids = [], [], []
            
            for j, (det, label) in enumerate(zip(dets, labels)):
                if len(det) >= 5:
                    x1, y1, x2, y2, score = det[:5]
                    
                    if score > self.score_threshold:
                        # Transform coordinates
                        scale = info['scale']
                        offset_x = info['offset_x']
                        offset_y = info['offset_y']
                        orig_w, orig_h = info['original_size']
                        
                        x1 = max(0, min(int((x1 - offset_x) / scale), orig_w))
                        y1 = max(0, min(int((y1 - offset_y) / scale), orig_h))
                        x2 = max(0, min(int((x2 - offset_x) / scale), orig_w))
                        y2 = max(0, min(int((y2 - offset_y) / scale), orig_h))
                        
                        if x2 > x1 and y2 > y1:
                            boxes.append((x1, y1, x2, y2))
                            scores.append(float(score))
                            class_ids.append(int(label))
            
            batch_results.append((boxes, scores, class_ids))
        
        return batch_results
    
    def draw_detections(self, image, boxes, scores, class_ids):
        """Draw detections on image"""
        if len(boxes) == 0:
            return image
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            color = colors[class_id % len(colors)]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            label = f"{class_name}: {score:.2f}"
            
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Sequential Batch Processing for Object Detection on Jetson Nano',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument('--mode', 
                       choices=['image', 'folder', 'video', 'live'], 
                       required=True,
                       help='Processing mode: single image, folder of images, video file, or live stream')
    
    # Input/Output arguments
    parser.add_argument('--input', '-i',
                       type=str,
                       help='Input path: image file, folder path, or video file (required for image/folder/video modes)')
    
    parser.add_argument('--output', '-o',
                       type=str,
                       help='Output path: image file, folder path, or video file (auto-generated if not specified)')
    
    parser.add_argument('--source', '-s',
                       type=str,
                       default='0',
                       help='Video source for live mode: camera index (0,1,2...) or stream URL (default: 0)')
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    
    model_group.add_argument('--model-path', '-m',
                           type=str,
                           help='Path to ONNX model file (auto-detected if not specified)')
    
    model_group.add_argument('--config', '-cfg',
                           type=str,
                           help='Path to configuration file (JSON/YAML) containing model and class info')
    
    model_group.add_argument('--classes', '-c',
                           nargs='+',
                           help='Class names for detection (space-separated)')
    
    model_group.add_argument('--classes-file',
                           type=str,
                           help='Path to file containing class names (one per line, JSON, or YAML)')
    
    model_group.add_argument('--preset-classes',
                           choices=['coco', 'vehicle', 'face', 'person'],
                           help='Use predefined class sets')
    
    model_group.add_argument('--list-models',
                           action='store_true',
                           help='List available model files and exit')
    
    model_group.add_argument('--model-search-paths',
                           nargs='+',
                           help='Additional paths to search for models')
    
    # Processing parameters
    parser.add_argument('--batch-size', '-b',
                       type=int,
                       default=1,
                       help='Batch size for processing (default: 4)')
    
    parser.add_argument('--input-size',
                       nargs=2,
                       type=int,
                       default=[640, 640],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Model input size in pixels (default: 640 640)')
    
    parser.add_argument('--score-threshold', '-t',
                       type=float,
                       default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    
    # Image processing options
    parser.add_argument('--image-extensions',
                       nargs='+',
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
                       help='Image file extensions to process (default: .jpg .jpeg .png .bmp .tiff .tif)')
    
    # Display settings
    parser.add_argument('--display-time',
                       type=int,
                       default=200,
                       help='Display time per frame in live mode (ms, default: 200)')
    
    return parser.parse_args()

def validate_arguments(args):
    """Validate command line arguments"""
    # Check model file exists
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Mode-specific validation
    if args.mode in ['image', 'folder', 'video']:
        if not args.input:
            raise ValueError(f"--input is required for {args.mode} mode")
        
        input_path = Path(args.input)
        
        if args.mode == 'image':
            if not input_path.exists():
                raise FileNotFoundError(f"Input image file not found: {args.input}")
            if not input_path.is_file():
                raise ValueError(f"Input path is not a file: {args.input}")
        
        elif args.mode == 'folder':
            if not input_path.exists():
                raise FileNotFoundError(f"Input folder not found: {args.input}")
            if not input_path.is_dir():
                raise ValueError(f"Input path is not a directory: {args.input}")
        
        elif args.mode == 'video':
            if not input_path.exists():
                raise FileNotFoundError(f"Input video file not found: {args.input}")
            if not input_path.is_file():
                raise ValueError(f"Input path is not a file: {args.input}")
    
    # Live mode validation
    if args.mode == 'live':
        # Convert source to int if it's a digit (camera index)
        if args.source.isdigit():
            args.source = int(args.source)
    
    # Validate parameters
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    if not (0.0 <= args.score_threshold <= 1.0):
        raise ValueError("Score threshold must be between 0.0 and 1.0")
    
    if any(size <= 0 for size in args.input_size):
        raise ValueError("Input size dimensions must be positive")
    
    return args

def main():
    """Main function with argument parsing"""
    try:
        # Parse and validate arguments
        args = parse_arguments()
        args = validate_arguments(args)
        
        print("=== Sequential Batch Processing for Object Detection ===")
        print(f"Mode: {args.mode}")
        print(f"Model: {args.model_path}")
        print(f"Classes: {args.classes}")
        print(f"Batch size: {args.batch_size}")
        print(f"Input size: {args.input_size[0]}x{args.input_size[1]}")
        print(f"Score threshold: {args.score_threshold}")
        print()
        
        # Setup ONNX session
        print("Loading model...")
        session = setup_jetson_session(args.model_path)
        print("Model loaded successfully!")
        
        # Create processor
        processor = SequentialBatchProcessor(
            session=session,
            batch_size=args.batch_size,
            target_size=tuple(args.input_size),
            class_names=args.classes,
            score_threshold=args.score_threshold
        )
        
        # Process based on mode
        if args.mode == 'image':
            print(f"Input image: {args.input}")
            output_path = args.output
            processor.process_single_image(args.input, output_path)
            
        elif args.mode == 'folder':
            print(f"Input folder: {args.input}")
            if args.output:
                print(f"Output folder: {args.output}")
            else:
                print(f"Output folder: {args.input}/detected_output")
            results = processor.process_image_folder(args.input, args.output, args.image_extensions)
            
        elif args.mode == 'video':
            output_path = args.output if args.output else './sequential_output.mp4'
            print(f"Input video: {args.input}")
            print(f"Output video: {output_path}")
            processor.process_video_sequential(args.input, output_path)
            print(f"Video processing complete! Output saved to: {output_path}")
            
        elif args.mode == 'live':
            print(f"Live source: {args.source}")
            processor.process_live_sequential(args.source)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == '__main__':
    exit(main())             