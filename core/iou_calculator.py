"""
IoU Calculator for Smart Hungarian - ByteTrack Style
====================================================

Lightweight IoU-based feature calculator untuk mengganti ReID extractor.
Menggunakan IoU calculation seperti ByteTrack untuk speed optimization.

FEATURES:
- IoU calculation antara tracks dan detections
- Distance calculation untuk spatial constraint
- Motion prediction untuk track prediction
- Lightweight dan fast (target 30+ FPS)
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import time

class IoUCalculator:
    """
    IoU Calculator seperti ByteTrack untuk Smart Hungarian
    
    Mengganti ReID extractor dengan IoU-based calculation yang lebih ringan
    """
    
    def __init__(self, config=None):
        """Initialize IoU Calculator"""
        # IoU calculation parameters - VERY STRICT untuk anti-ghost
        self.iou_threshold = 0.3  # Minimum IoU untuk association
        self.distance_threshold = 80.0  # Maximum distance (VERY STRICT - was 100)
        
        # Motion prediction parameters
        self.motion_weight = 0.7  # Weight untuk motion prediction
        self.max_motion_error = 50.0  # Maximum motion prediction error (VERY STRICT - was 60)
        
        # Performance tracking
        self.calculation_times = []
        
        print("✓ IoU Calculator initialized (ByteTrack style)")
        print(f"  IoU threshold: {self.iou_threshold}")
        print(f"  Distance threshold: {self.distance_threshold}")
        print(f"  Motion weight: {self.motion_weight}")
    
    def calculate_iou_matrix(self, tracks: List, detections: List[Dict]) -> np.ndarray:
        """
        Calculate IoU matrix antara tracks dan detections
        
        Args:
            tracks: List of track objects
            detections: List of detection dictionaries
            
        Returns:
            IoU matrix [N_tracks, N_detections]
        """
        start_time = time.time()
        
        n_tracks = len(tracks)
        n_detections = len(detections)
        
        if n_tracks == 0 or n_detections == 0:
            return np.zeros((n_tracks, n_detections), dtype=np.float32)
        
        iou_matrix = np.zeros((n_tracks, n_detections), dtype=np.float32)
        
        for i, track in enumerate(tracks):
            track_bbox = self._get_track_bbox(track)
            
            for j, detection in enumerate(detections):
                det_bbox = detection['bbox']
                
                # Calculate IoU
                iou = self._compute_iou(track_bbox, det_bbox)
                iou_matrix[i, j] = iou
        
        calculation_time = time.time() - start_time
        self.calculation_times.append(calculation_time)
        
        return iou_matrix
    
    def calculate_distance_matrix(self, tracks: List, detections: List[Dict]) -> np.ndarray:
        """
        Calculate distance matrix antara track centers dan detection centers
        
        Args:
            tracks: List of track objects
            detections: List of detection dictionaries
            
        Returns:
            Distance matrix [N_tracks, N_detections]
        """
        n_tracks = len(tracks)
        n_detections = len(detections)
        
        if n_tracks == 0 or n_detections == 0:
            return np.zeros((n_tracks, n_detections), dtype=np.float32)
        
        distance_matrix = np.zeros((n_tracks, n_detections), dtype=np.float32)
        
        for i, track in enumerate(tracks):
            track_center = self._get_track_center(track)
            
            for j, detection in enumerate(detections):
                det_center = detection['center']
                
                # Calculate Euclidean distance
                distance = np.sqrt(
                    (track_center[0] - det_center[0])**2 + 
                    (track_center[1] - det_center[1])**2
                )
                distance_matrix[i, j] = distance
        
        return distance_matrix
    
    def calculate_motion_matrix(self, tracks: List, detections: List[Dict]) -> np.ndarray:
        """
        Calculate motion prediction matrix
        
        Args:
            tracks: List of track objects
            detections: List of detection dictionaries
            
        Returns:
            Motion error matrix [N_tracks, N_detections]
        """
        n_tracks = len(tracks)
        n_detections = len(detections)
        
        if n_tracks == 0 or n_detections == 0:
            return np.zeros((n_tracks, n_detections), dtype=np.float32)
        
        motion_matrix = np.zeros((n_tracks, n_detections), dtype=np.float32)
        
        for i, track in enumerate(tracks):
            # Get track velocity/prediction
            predicted_center = self._predict_track_position(track)
            
            for j, detection in enumerate(detections):
                det_center = detection['center']
                
                # Calculate motion prediction error
                motion_error = np.sqrt(
                    (predicted_center[0] - det_center[0])**2 + 
                    (predicted_center[1] - det_center[1])**2
                )
                motion_matrix[i, j] = motion_error
        
        return motion_matrix
    
    def calculate_combined_features(self, tracks: List, detections: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Calculate combined IoU-based features seperti ByteTrack
        
        Args:
            tracks: List of track objects
            detections: List of detection dictionaries
            
        Returns:
            Dictionary dengan berbagai feature matrices
        """
        start_time = time.time()
        
        # Calculate individual matrices
        iou_matrix = self.calculate_iou_matrix(tracks, detections)
        distance_matrix = self.calculate_distance_matrix(tracks, detections)
        motion_matrix = self.calculate_motion_matrix(tracks, detections)
        
        # Normalize matrices
        normalized_distance = np.clip(distance_matrix / self.distance_threshold, 0, 1)
        normalized_motion = np.clip(motion_matrix / self.max_motion_error, 0, 1)
        
        # Calculate combined cost matrix (lower is better)
        # ANTI-GHOST: Stricter weights untuk distance & motion
        cost_matrix = (
            1.0 - iou_matrix +                    # IoU cost (higher IoU = lower cost)
            0.3 * normalized_distance +           # Distance cost (INCREASED from 0.1)
            0.2 * normalized_motion               # Motion prediction cost (INCREASED from 0.05)
        )
        
        # CRITICAL: Validate matrix dimensions
        expected_shape = (len(tracks), len(detections))
        if cost_matrix.shape != expected_shape:
            logger.warning(f"IoU Calculator dimension mismatch - rebuilding matrix!")
            logger.warning(f"   Expected: {expected_shape}")
            logger.warning(f"   Got: {cost_matrix.shape}")
            logger.warning(f"   Tracks: {len(tracks)}, Detections: {len(detections)}")
            
            # Rebuild ALL matrices with correct dimensions
            iou_matrix = np.zeros(expected_shape, dtype=np.float32)
            distance_matrix = np.zeros(expected_shape, dtype=np.float32)
            motion_matrix = np.zeros(expected_shape, dtype=np.float32)
            
            # Recalculate with correct dimensions
            for i in range(len(tracks)):
                track_bbox = self._get_track_bbox(tracks[i])
                track_center = self._get_track_center(tracks[i])
                
                for j in range(len(detections)):
                    det_bbox = detections[j]['bbox']
                    det_center = detections[j]['center']
                    
                    # Recalculate IoU
                    iou = self._compute_iou(track_bbox, det_bbox)
                    iou_matrix[i, j] = iou
                    
                    # Recalculate distance
                    distance = np.sqrt(
                        (track_center[0] - det_center[0])**2 + 
                        (track_center[1] - det_center[1])**2
                    )
                    distance_matrix[i, j] = distance
                    
                    # Recalculate motion
                    predicted_center = self._predict_track_position(tracks[i])
                    motion_error = np.sqrt(
                        (predicted_center[0] - det_center[0])**2 + 
                        (predicted_center[1] - det_center[1])**2
                    )
                    motion_matrix[i, j] = motion_error
            
            # Recalculate normalized matrices
            normalized_distance = np.clip(distance_matrix / self.distance_threshold, 0, 1)
            normalized_motion = np.clip(motion_matrix / self.max_motion_error, 0, 1)
            
            # Recalculate cost matrix (same weights as above)
            cost_matrix = (
                1.0 - iou_matrix +
                0.3 * normalized_distance +
                0.2 * normalized_motion
            )
        
        calculation_time = time.time() - start_time
        self.calculation_times.append(calculation_time)
        
        return {
            'iou_matrix': iou_matrix,
            'distance_matrix': distance_matrix,
            'motion_matrix': motion_matrix,
            'cost_matrix': cost_matrix,
            'calculation_time': calculation_time
        }
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Compute IoU between two bounding boxes [x1, y1, x2, y2]
        """
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            
            # Union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    def _get_track_bbox(self, track) -> List[float]:
        """Get track bounding box"""
        if hasattr(track, 'bbox'):
            return track.bbox
        elif hasattr(track, 'current_detection') and 'bbox' in track.current_detection:
            return track.current_detection['bbox']
        else:
            return [0, 0, 0, 0]
    
    def _get_track_center(self, track) -> List[float]:
        """Get track center coordinates"""
        if hasattr(track, 'center'):
            return track.center
        elif hasattr(track, 'current_detection') and 'center' in track.current_detection:
            return track.current_detection['center']
        else:
            bbox = self._get_track_bbox(track)
            return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    
    def _predict_track_position(self, track) -> List[float]:
        """Predict track position untuk motion matrix"""
        current_center = self._get_track_center(track)
        
        # Simple motion prediction menggunakan velocity
        if hasattr(track, 'velocity') and track.velocity is not None:
            predicted_center = [
                current_center[0] + track.velocity[0],
                current_center[1] + track.velocity[1]
            ]
        else:
            # No motion prediction, return current position
            predicted_center = current_center
        
        return predicted_center
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.calculation_times:
            return {'available': False}
        
        return {
            'available': True,
            'total_calculations': len(self.calculation_times),
            'avg_calculation_time': np.mean(self.calculation_times),
            'min_calculation_time': np.min(self.calculation_times),
            'max_calculation_time': np.max(self.calculation_times),
            'total_time': np.sum(self.calculation_times)
        }
    
    def visualize_iou_matrix(self, iou_matrix: np.ndarray, tracks: List, detections: List[Dict]) -> np.ndarray:
        """
        Visualize IoU matrix untuk debugging
        """
        if iou_matrix.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create visualization
        height, width = iou_matrix.shape
        vis_image = np.zeros((height * 50, width * 50, 3), dtype=np.uint8)
        
        for i in range(height):
            for j in range(width):
                iou_value = iou_matrix[i, j]
                
                # Color based on IoU value
                if iou_value > 0.5:
                    color = (0, 255, 0)  # Green for high IoU
                elif iou_value > 0.3:
                    color = (0, 255, 255)  # Yellow for medium IoU
                else:
                    color = (0, 0, 255)  # Red for low IoU
                
                # Draw rectangle
                y1, y2 = i * 50, (i + 1) * 50
                x1, x2 = j * 50, (j + 1) * 50
                vis_image[y1:y2, x1:x2] = color
                
                # Add text
                text = f"{iou_value:.2f}"
                cv2.putText(vis_image, text, (x1 + 5, y1 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return vis_image


# Testing function
if __name__ == "__main__":
    print("Testing IoU Calculator (ByteTrack style)")
    
    # Create test data
    tracks = []
    for i in range(3):
        track = type('Track', (), {
            'track_id': i,
            'bbox': [100 + i*50, 100 + i*30, 150 + i*50, 150 + i*30],
            'center': [125 + i*50, 125 + i*30],
            'velocity': [2.0, 1.0]
        })()
        tracks.append(track)
    
    detections = [
        {'bbox': [105, 105, 155, 155], 'center': [130, 130]},
        {'bbox': [155, 135, 205, 185], 'center': [180, 160]},
        {'bbox': [205, 165, 255, 215], 'center': [230, 190]},
        {'bbox': [255, 195, 305, 245], 'center': [280, 220]}
    ]
    
    # Test IoU Calculator
    iou_calc = IoUCalculator()
    
    # Calculate features
    features = iou_calc.calculate_combined_features(tracks, detections)
    
    print(f"IoU Matrix shape: {features['iou_matrix'].shape}")
    print(f"IoU Matrix:\n{features['iou_matrix']}")
    print(f"Cost Matrix:\n{features['cost_matrix']}")
    print(f"Calculation time: {features['calculation_time']:.4f}s")
    
    # Performance stats
    stats = iou_calc.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("✓ IoU Calculator test completed")

