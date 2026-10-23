"""
YOLOv8x Detector for High-Accuracy Pedestrian Detection in GBC-MOT POC
Uses centralized settings from utils.settings
"""

import torch
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
from pathlib import Path

# Try import ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ ultralytics not installed. Please run: pip install ultralytics")

class YOLOv8Detector:
    """
    YOLOv8x-based high-accuracy pedestrian detector
    Optimized untuk detection quality pada complex scenes
    """
    
    def __init__(self, **kwargs):

        """
        Initialize YOLOv8x detector using centralized settings
        """
        from utils.settings import SETTINGS
        
        # Get config from centralized settings
        config = SETTINGS.get_detector_config()
        
        # Override with any kwargs provided
        for key, value in kwargs.items():
            if key in config:
                config[key] = value
        
        # Apply configuration
        self.model_variant = config['model_variant']
        self.confidence_threshold = config['confidence_threshold']
        self.nms_threshold = config['nms_threshold']
        self.device = config['device']
        self.input_size = config['input_size']
        self.half_precision = config['half_precision']
        self.fp16_active = False
        
        # Load model — class names dibaca otomatis dari weight, tidak hardcoded
        self.model = self._load_model()

        # Setelah model loaded, pakai class names langsung dari weight file
        # model.names → dict {0: 'car', 1: 'truck', ...} (apapun weightnya)
        self.model_names: dict = {int(k): str(v) for k, v in self.model.names.items()}

        # Jika settings menyediakan target_classes eksplisit, filter dari model_names.
        # Jika tidak (None / kosong), gunakan SEMUA class dari weight.
        _cfg_classes = config.get('target_classes', None)
        if _cfg_classes:
            # Filter: hanya pakai class yang ada DI model DAN ada di cfg list
            self.target_classes = [
                name for name in self.model_names.values()
                if name.lower() in [c.lower() for c in _cfg_classes]
            ]
            self.class_ids = [
                cid for cid, name in self.model_names.items()
                if name.lower() in [c.lower() for c in _cfg_classes]
            ]
        else:
            # Semua class dari weight — tidak ada hardcoding
            self.target_classes = list(self.model_names.values())
            self.class_ids = list(self.model_names.keys())
        
        # Performance tracking
        self.detection_times = []
        self.total_detections = 0
        
        print(f"✓ Detector initialized on {self.device}")
        print(f"  Weight   : {self.model_variant}")
        print(f"  Classes  : {len(self.model_names)} total in weight → "
              f"using {len(self.class_ids)}: {self.target_classes}")
        print(f"  Input size: {self.input_size}  |  Conf: {self.confidence_threshold}")
        
    def _load_model(self) -> YOLO:
        """Load YOLOv8x model dengan optimasi."""
        try:
            model = YOLO(self.model_variant)

            if self.device == 'cuda':
                model.to(self.device)
                if self.half_precision:
                    try:
                        model.half()
                        self.fp16_active = True
                        print("FP16 ACTIVE: detector running in half precision")
                    except Exception as e:
                        raise RuntimeError(f"FP16 requested but activation failed: {e}") from e
                else:
                    print("FP16 INACTIVE: detector running in FP32")
            else:
                print("FP16 INACTIVE: CPU detector uses FP32")

            model.model.eval()

            dummy_input = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            try:
                for _ in range(5):
                    _ = model(dummy_input, verbose=False)
            except RuntimeError as e:
                if self.half_precision:
                    print(f"FP16 warmup failed, falling back to FP32: {e}")
                    model.float()
                    self.fp16_active = False
                    for _ in range(5):
                        _ = model(dummy_input, verbose=False)
                    print("FP16 FALLBACK: detector running in FP32")
                else:
                    raise

            print("YOLOv8x warmup completed")
            if self.fp16_active:
                print("FP16 CONFIRMED after warmup")

            return model

        except Exception as e:
            print(f"Error loading YOLOv8x model: {e}")
            raise

    def detect(self, image: np.ndarray, return_stats: bool = False) -> List[Dict]:
        """
        Detect pedestrians dalam image dengan YOLOv8x
        
        Args:
            image: Input image (H, W, C) dalam BGR format
            return_stats: Whether to return detection statistics
            
        Returns:
            List of detection dictionaries
        """
        start_time = time.time()
        
        # YOLOv8x inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            classes=self.class_ids,  # Dynamic class IDs
            verbose=False,
            imgsz=self.input_size
        )
        
        # Process hasil
        detections = self._process_results(results[0], image.shape[:2])
        
        # Track performance
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        self.total_detections += len(detections)
        
        if return_stats:
            stats = {
                'detection_time': detection_time,
                'num_detections': len(detections),
                'input_size': self.input_size,
                'model_variant': self.model_variant
            }
            return detections, stats
        
        return detections
    
    def _process_results(self, result, image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Process YOLOv8x results ke format detection standard
        
        Args:
            result: YOLOv8x result object
            image_shape: (height, width) dari input image
            
        Returns:
            List of processed detections
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Extract boxes, scores, classes
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for i in range(len(boxes)):
            class_id = int(classes[i])

            # Filter hanya target classes (sudah dihitung dari model.names di __init__)
            if class_id not in self.class_ids:
                continue

            bbox = boxes[i]
            confidence = scores[i]
            # Nama class dibaca dari weight, bukan hardcoded
            class_name = self.model_names.get(class_id, f'class_{class_id}')
            
            # Extract coordinates
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Clamp ke image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_shape[1], x2)
            y2 = min(image_shape[0], y2)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Calculate center dan size
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1
            
            # Basic size filtering (adaptive per class)
            min_width = 8 if class_name == 'person' else 15
            min_height = 15 if class_name == 'person' else 15
            
            if width < min_width or height < min_height:
                continue
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'center': [center_x, center_y],
                'size': [width, height],
                'aspect_ratio': height / width if width > 0 else 0,
                'class_id': class_id,
                'class_name': class_name
            }
            
            detections.append(detection)
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Batch detection untuk multiple images (optimal untuk accuracy)
        
        Args:
            images: List of input images
            
        Returns:
            List of detection lists untuk each image
        """
        start_time = time.time()
        
        # YOLOv8x mendukung batch inference
        results = self.model(
            images,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            classes=self.class_ids,  # Dynamic class IDs
            verbose=False,
            imgsz=self.input_size
        )
        
        # Process semua results
        batch_detections = []
        for i, result in enumerate(results):
            image_shape = images[i].shape[:2]
            detections = self._process_results(result, image_shape)
            batch_detections.append(detections)
        
        # Track performance
        batch_time = time.time() - start_time
        total_detections = sum(len(dets) for dets in batch_detections)
        
        self.detection_times.append(batch_time)
        self.total_detections += total_detections
        
        return batch_detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], 
                           show_fps: bool = True, show_stats: bool = True) -> np.ndarray:
        """
        Visualize detections dengan performance info
        
        Args:
            image: Input image
            detections: List of detections
            show_fps: Whether to show FPS info
            
        Returns:
            Image dengan drawn bounding boxes
        """
        vis_image = image.copy()
        
        # Color mapping per class
        class_colors = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),         # Blue
            'motorcycle': (0, 165, 255), # Orange
            'bus': (255, 255, 0),       # Cyan
            'truck': (255, 0, 255)      # Magenta
        }
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection.get('class_name', 'unknown')
            
            x1, y1, x2, y2 = bbox
            
            # Get color based on class
            base_color = class_colors.get(class_name, (128, 128, 128))
            
            # Adjust brightness based on confidence
            if confidence > 0.7:
                color = base_color
            elif confidence > 0.5:
                color = tuple(int(c * 0.8) for c in base_color)
            else:
                color = tuple(int(c * 0.6) for c in base_color)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with class name
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Show performance info
        if show_fps and self.detection_times:
            avg_time = np.mean(self.detection_times[-10:])  # Last 10 frames
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Count per class
            class_counts = {}
            for det in detections:
                cls = det.get('class_name', 'unknown')
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            counts_str = ', '.join([f"{k}:{v}" for k, v in class_counts.items()])
            perf_text = f"YOLOv8x: {len(detections)} dets ({counts_str}), {fps:.1f} FPS"
            cv2.putText(vis_image, perf_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        if not self.detection_times:
            return {
                'avg_detection_time': 0,
                'avg_fps': 0,
                'total_detections': 0,
                'model_info': self.get_model_info()
            }
        
        avg_time = np.mean(self.detection_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_detection_time': avg_time,
            'avg_fps': avg_fps,
            'min_time': np.min(self.detection_times),
            'max_time': np.max(self.detection_times),
            'total_detections': self.total_detections,
            'total_inferences': len(self.detection_times),
            'detections_per_inference': self.total_detections / len(self.detection_times),
            'model_info': self.get_model_info()
        }
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        # Estimate model size
        model_size_mb = 68.0 if 'x' in self.model_variant else 6.0  # YOLOv8x size
        
        return {
            'model_name': f'YOLOv8x',
            'model_variant': self.model_variant,
            'device': self.device,
            'input_size': self.input_size,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'half_precision': self.half_precision,
            'target_classes': self.target_classes,
            'class_ids': self.class_ids,
            'estimated_size_mb': model_size_mb,
            'optimization_level': 'high_accuracy'
        }
    
    def optimize_for_accuracy(self):
        """Apply accuracy optimizations"""
        print("🎯 Applying accuracy optimizations...")
        
        # Set optimal batch size
        if hasattr(self.model, 'model'):
            try:
                # Enable optimized operations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print("✓ CUDNN optimizations enabled")
            except:
                pass
        
        # Additional warmup for larger model
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(5):  # More warmup iterations for YOLOv8x
            _ = self.detect(dummy_image)
        
        print("✓ YOLOv8x warmup completed")
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.detection_times = []
        self.total_detections = 0


# Backward compatibility - alias untuk SSDDetector
class SSDDetector(YOLOv8Detector):
    """Backward compatibility alias"""
    def __init__(self, **kwargs):
        # Map SSD parameters ke YOLOv8x parameters
        if 'confidence_threshold' not in kwargs:
            kwargs['confidence_threshold'] = 0.2  # Optimized for recall
        if 'nms_threshold' not in kwargs:
            kwargs['nms_threshold'] = 0.3  # Optimized for recall
            
        super().__init__(**kwargs)
        print("🔄 Using YOLOv8x as SSD replacement for better accuracy")


# Example usage dan testing
if __name__ == "__main__":
    # Test YOLOv8x detector
    detector = YOLOv8Detector(confidence_threshold=0.2, nms_threshold=0.3)
    
    # Print model info
    info = detector.get_model_info()
    print("YOLOv8x Detector Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Apply optimizations
    detector.optimize_for_accuracy()
    
    # Test dengan dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    print("\nTesting YOLOv8x detection accuracy...")
    
    # Benchmark speed
    times = []
    for i in range(10):
        start = time.time()
        detections = detector.detect(dummy_image)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"Average detection time: {avg_time:.4f}s")
    print(f"Estimated FPS: {fps:.1f}")
    print(f"Found {len(detections)} detections in test")
    
    # Performance stats
    perf_stats = detector.get_performance_stats()
    print(f"Performance stats: {perf_stats}")
