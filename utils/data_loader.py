"""
MOT17 Dataset Loader for GBC-MOT POC
Uses centralized settings from utils.settings
Loads sequence images and ground truth annotations from MOT17 dataset
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Iterator
from pathlib import Path
import glob

class MOT17Reader:
    """
    MOT17 dataset reader for multi-object tracking
    Handles sequence loading, ground truth parsing, and frame-by-frame iteration
    """
    
    def __init__(self, 
             dataset_path: str = None,
             sequence_name: str = None,
             load_gt: bool = True):
        """
        Initialize MOT17 dataset reader using centralized settings
        
        Args:
            dataset_path: Path to MOT17 dataset root (uses centralized settings if None)
            sequence_name: Name of sequence to load (uses centralized settings if None)
            load_gt: Whether to load ground truth annotations
        """
        from utils.settings import SETTINGS
    
        # Use centralized settings if parameters not provided
        if dataset_path is None or sequence_name is None:
            dataset_info = SETTINGS.get_dataset_info()
            if dataset_path is None:
                dataset_path = dataset_info['dataset_root']
            if sequence_name is None:
                sequence_name = dataset_info['sequence_name']

        self.dataset_path = Path(dataset_path)
        self.sequence_name = sequence_name
        self.load_gt = load_gt

        # Use pre-validated paths from settings if available
        try:
            dataset_info = SETTINGS.get_dataset_info()
            if (str(self.dataset_path) == dataset_info['dataset_root'] and 
                self.sequence_name == dataset_info['sequence_name']):
                # Use pre-validated paths from centralized settings
                self.sequence_path = Path(dataset_info['sequence_path'])
                self.images_path = Path(dataset_info['images_path'])
                self.gt_path = Path(dataset_info['gt_path'])
                path_detection_needed = False
            else:
                # Manual path detection for custom parameters
                path_detection_needed = True
        except:
            # Fallback to manual detection
            path_detection_needed = True

        if path_detection_needed:
            # Paths - check multiple possible structures
            possible_sequence_paths = [
                self.dataset_path / "train" / sequence_name,  # Standard MOT17 structure
                self.dataset_path / sequence_name             # Direct structure
            ]
            
            self.sequence_path = None
            for path in possible_sequence_paths:
                if path.exists():
                    self.sequence_path = path
                    break
            
            if self.sequence_path is None:
                raise FileNotFoundError(f"Sequence not found in any expected location: {possible_sequence_paths}")
            
            self.images_path = self.sequence_path / "img1"
            self.gt_path = self.sequence_path / "gt" / "gt.txt"
        
        # Validate paths
        self._validate_paths()
        
        # Load sequence info
        self.sequence_info = self._load_sequence_info()
        self.image_files = self._get_image_files()
        
        # Load ground truth if requested
        self.gt_data = self._load_ground_truth() if load_gt else None
        
        print(f"✓ MOT17 sequence loaded: {sequence_name}")
        print(f"  Images: {len(self.image_files)}")
        print(f"  Resolution: {self.sequence_info.get('imWidth', 'Unknown')}x{self.sequence_info.get('imHeight', 'Unknown')}")
        print(f"  FPS: {self.sequence_info.get('frameRate', 'Unknown')}")
        if self.gt_data is not None:
            print(f"  GT annotations: {len(self.gt_data)} detections")
    
    def _validate_paths(self):
        """Validate that all required paths exist"""
        if self.sequence_path is None:
            raise FileNotFoundError(f"Sequence not found in any expected location")
        
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images path not found: {self.images_path}")
        
        if self.load_gt and not self.gt_path.exists():
            print(f"Warning: Ground truth file not found: {self.gt_path}")
            self.load_gt = False
            
        print(f"✓ Dataset paths validated:")
        print(f"  Sequence: {self.sequence_path}")
        print(f"  Images: {self.images_path}")
        print(f"  GT: {self.gt_path} ({'Found' if self.gt_path.exists() else 'Not found'})")
    
    def _load_sequence_info(self) -> Dict:
        """Load sequence information from seqinfo.ini"""
        seqinfo_path = self.sequence_path / "seqinfo.ini"
        sequence_info = {}
        
        if seqinfo_path.exists():
            with open(seqinfo_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('['):
                        key, value = line.split('=', 1)
                        # Try to convert to appropriate type
                        try:
                            if '.' in value:
                                sequence_info[key] = float(value)
                            else:
                                sequence_info[key] = int(value)
                        except ValueError:
                            sequence_info[key] = value
        
        return sequence_info
    
    def _get_image_files(self) -> List[Path]:
        """Get sorted list of image files"""
        # Common image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        image_files = []
        for ext in extensions:
            image_files.extend(self.images_path.glob(ext))
        
        # Sort by frame number (extracted from filename)
        image_files.sort(key=lambda x: int(x.stem))
        
        return image_files
    
    def _load_ground_truth(self) -> Optional[pd.DataFrame]:
        """
        Load ground truth annotations from gt.txt
        
        Returns:
            DataFrame with columns: frame_id, track_id, x, y, w, h, conf, class_id, visibility
        """
        if not self.gt_path.exists():
            return None
        
        try:
            # MOT format: frame, id, bbox_left, bbox_top, bbox_w, bbox_h, conf, x, y, z
            columns = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'conf', 'class_id', 'visibility']
            
            gt_data = pd.read_csv(self.gt_path, header=None, names=columns)
            
            # Filter for pedestrians (class_id == 1) and valid tracks (conf == 1)
            gt_data = gt_data[(gt_data['class_id'].isin([1, 2, 7])) & (gt_data['conf'] == 1)]
            
            # Convert coordinates to integers
            for col in ['frame_id', 'track_id', 'x', 'y', 'w', 'h']:
                gt_data[col] = gt_data[col].astype(int)
            
            return gt_data
            
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            return None
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load specific frame by index
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            Image as numpy array (BGR format) or None if not found
        """
        if frame_idx < 0 or frame_idx >= len(self.image_files):
            return None
        
        image_path = self.image_files[frame_idx]
        image = cv2.imread(str(image_path))
        
        return image
    
    def get_frame_by_id(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Load frame by MOT frame ID (1-based)
        
        Args:
            frame_id: MOT frame ID (1-based)
            
        Returns:
            Image as numpy array (BGR format) or None if not found
        """
        frame_idx = frame_id - 1  # Convert to 0-based index
        return self.get_frame(frame_idx)
    
    def get_frame_annotations(self, frame_id: int) -> List[Dict]:
        """
        Get ground truth annotations for specific frame
        
        Args:
            frame_id: MOT frame ID (1-based)
            
        Returns:
            List of annotation dictionaries with keys:
            - 'track_id': Ground truth track ID
            - 'bbox': [x, y, w, h] in MOT format
            - 'center': [cx, cy] center point
            - 'visibility': Visibility score
        """
        if self.gt_data is None:
            return []
        
        frame_gt = self.gt_data[self.gt_data['frame_id'] == frame_id]
        
        annotations = []
        for _, row in frame_gt.iterrows():
            x, y, w, h = row['x'], row['y'], row['w'], row['h']
            
            annotation = {
                'track_id': int(row['track_id']),
                'bbox': [x, y, w, h],
                'bbox_xyxy': [x, y, x + w, y + h],  # Convert to xyxy format
                'center': [x + w/2, y + h/2],
                'visibility': float(row['visibility'])
            }
            
            annotations.append(annotation)
        
        return annotations
    
    def get_sequence_length(self) -> int:
        """Get total number of frames in sequence"""
        return len(self.image_files)
    
    def get_frame_range(self) -> Tuple[int, int]:
        """
        Get frame ID range (MOT format, 1-based)
        
        Returns:
            Tuple of (start_frame_id, end_frame_id)
        """
        if not self.image_files:
            return (1, 1)
        
        start_id = int(self.image_files[0].stem)
        end_id = int(self.image_files[-1].stem)
        
        return (start_id, end_id)
    
    def iterate_frames(self, start_frame: Optional[int] = None, 
                      end_frame: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray, List[Dict]]]:
        """
        Iterate through frames with annotations
        
        Args:
            start_frame: Starting frame ID (1-based, inclusive)
            end_frame: Ending frame ID (1-based, inclusive)
            
        Yields:
            Tuple of (frame_id, image, annotations)
        """
        frame_start, frame_end = self.get_frame_range()
        
        if start_frame is None:
            start_frame = frame_start
        if end_frame is None:
            end_frame = frame_end
        
        for frame_id in range(start_frame, end_frame + 1):
            image = self.get_frame_by_id(frame_id)
            if image is None:
                continue
            
            annotations = self.get_frame_annotations(frame_id)
            
            yield frame_id, image, annotations
    
    def get_track_info(self) -> Dict:
        """
        Get information about tracks in the sequence
        
        Returns:
            Dictionary with track statistics
        """
        if self.gt_data is None:
            return {}
        
        track_ids = self.gt_data['track_id'].unique()
        track_lengths = self.gt_data['track_id'].value_counts()
        
        return {
            'num_tracks': len(track_ids),
            'track_ids': sorted(track_ids.tolist()),
            'avg_track_length': track_lengths.mean(),
            'min_track_length': track_lengths.min(),
            'max_track_length': track_lengths.max(),
            'total_detections': len(self.gt_data)
        }
    
    def visualize_frame(self, frame_id: int, 
                       show_gt: bool = True,
                       show_track_ids: bool = True) -> np.ndarray:
        """
        Visualize frame with annotations for debugging
        
        Args:
            frame_id: Frame ID to visualize
            show_gt: Whether to show ground truth boxes
            show_track_ids: Whether to show track IDs
            
        Returns:
            Annotated image
        """
        image = self.get_frame_by_id(frame_id)
        if image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        vis_image = image.copy()
        
        if show_gt:
            annotations = self.get_frame_annotations(frame_id)
            
            for ann in annotations:
                bbox_xyxy = ann['bbox_xyxy']
                track_id = ann['track_id']
                visibility = ann['visibility']
                
                x1, y1, x2, y2 = bbox_xyxy
                
                # Color based on track ID
                color = self._get_track_color(track_id)
                
                # Draw bounding box
                thickness = 2 if visibility > 0.5 else 1
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
                
                # Draw track ID
                if show_track_ids:
                    label = f"ID:{track_id}"
                    if visibility < 1.0:
                        label += f" v:{visibility:.1f}"
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 5), 
                                 (x1 + label_size[0], y1), color, -1)
                    cv2.putText(vis_image, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add frame info
        frame_info = f"Frame: {frame_id}"
        cv2.putText(vis_image, frame_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return vis_image
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for track ID"""
        # Simple hash-based color generation
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color
    
    def get_dataset_info(self) -> Dict:
        """Get comprehensive dataset information"""
        info = {
            'sequence_name': self.sequence_name,
            'dataset_path': str(self.dataset_path),
            'num_frames': len(self.image_files),
            'sequence_info': self.sequence_info,
            'has_gt': self.gt_data is not None
        }
        
        if self.gt_data is not None:
            info.update(self.get_track_info())
        
        return info


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset loader
    dataset_path = "data/MOT17"  # Adjust path as needed
    
    try:
        # Load sequence
        reader = MOT17Reader(dataset_path, "MOT17-02-SDP", load_gt=True)
        
        # Print dataset info
        info = reader.get_dataset_info()
        print("\nDataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test frame loading
        print(f"\nTesting frame loading...")
        frame_id = 1
        image = reader.get_frame_by_id(frame_id)
        if image is not None:
            print(f"Frame {frame_id} loaded: {image.shape}")
            
            # Get annotations
            annotations = reader.get_frame_annotations(frame_id)
            print(f"Frame {frame_id} annotations: {len(annotations)}")
            
            if annotations:
                print("Sample annotation:")
                print(annotations[0])
        
        # Test iteration
        print(f"\nTesting frame iteration (first 3 frames)...")
        frame_count = 0
        for frame_id, image, annotations in reader.iterate_frames():
            print(f"Frame {frame_id}: {image.shape}, {len(annotations)} annotations")
            frame_count += 1
            if frame_count >= 3:
                break
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure MOT17 dataset is available at the specified path")
    except Exception as e:
        print(f"Error: {e}")