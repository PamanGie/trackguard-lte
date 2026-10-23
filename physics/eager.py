"""
EAGER: Energy-Aware Smoothing for LTE-TrackGuard
================================================

Adaptasi dari LTE-Face EAGER framework untuk traffic domain.
Mengurangi YOLO bounding box jitter sebelum physics calculations.

From Blueprint Section 2.3: EAGER Smoothing
"""

import numpy as np
from typing import Dict, List, Tuple
import time


class EAGERSmoother:
    """
    Energy-based trajectory stabilization untuk mengatasi YOLO jitter
    
    Implements energy minimization dari Blueprint Section 2.3
    """
    
    def __init__(self, config: Dict):
        """
        Initialize EAGER smoother
        
        Args:
            config: Configuration dari PHYSICS_CONFIG['eager']
        """
        self.config = config
        
        # Energy weights dari blueprint
        self.alpha = config.get('alpha', 0.2)  # EMA smoothing factor
        self.w_internal = config.get('w_internal', 0.3)
        self.w_external = config.get('w_external', 0.8)
        self.k_temp = config.get('k_temp', 0.5)
        self.iterations = config.get('iterations', 5)
        
        # Smoothed state storage per track
        # Format: {track_id: {'x_smooth': array, 'ar_smooth': float}}
        self.smoothed_states = {}
        
        # Performance tracking
        self.total_smooths = 0
        self.avg_jitter_reduction = 0.0
        
        print("✓ EAGER Smoother initialized")
        print(f"  Alpha (EMA): {self.alpha}")
        print(f"  Iterations: {self.iterations}")
        print(f"  Weights: internal={self.w_internal}, external={self.w_external}, temp={self.k_temp}")
    
    def smooth_detection(self, track_id: int, detection: Dict) -> Dict:
        """
        Smooth single detection untuk reduce jitter
        
        Args:
            track_id: Track ID
            detection: Detection dict dengan 'bbox', 'center'
            
        Returns:
            Smoothed detection dict
        """
        bbox = detection['bbox']
        center = detection['center']
        
        # Initialize smoothed state untuk track baru
        if track_id not in self.smoothed_states:
            self.smoothed_states[track_id] = {
                'x_smooth': np.array(center, dtype=np.float32),
                'ar_smooth': self._compute_aspect_ratio(bbox),
                'prev_center': np.array(center, dtype=np.float32),
                'prev_prev_center': np.array(center, dtype=np.float32)
            }
            return detection  # First frame, no smoothing
        
        state = self.smoothed_states[track_id]
        
        # Current raw position
        x_detect = np.array(center, dtype=np.float32)
        ar_detect = self._compute_aspect_ratio(bbox)
        
        # Energy-based optimization (Blueprint Section 2.3.2)
        x_smooth = self._optimize_position(
            x_detect, 
            state['x_smooth'],
            state['prev_center'],
            state['prev_prev_center']
        )
        
        # Aspect ratio smoothing (Blueprint Eq. 9)
        ar_smooth = self._smooth_aspect_ratio(ar_detect, state['ar_smooth'])
        
        # Update state
        state['prev_prev_center'] = state['prev_center'].copy()
        state['prev_center'] = state['x_smooth'].copy()
        state['x_smooth'] = x_smooth
        state['ar_smooth'] = ar_smooth
        
        # Create smoothed detection
        smoothed_detection = detection.copy()
        smoothed_detection['center'] = x_smooth.tolist()
        smoothed_detection['bbox'] = self._reconstruct_bbox(x_smooth, bbox, ar_smooth)
        smoothed_detection['smoothed'] = True
        
        self.total_smooths += 1
        
        return smoothed_detection
    
    def _optimize_position(self, x_detect: np.ndarray, x_prev: np.ndarray,
                          x_t_minus_1: np.ndarray, x_t_minus_2: np.ndarray) -> np.ndarray:
        """
        Optimize position menggunakan energy minimization (Blueprint Eq. 8-12)
        
        E_total = E_internal + E_external + E_temporal
        
        Args:
            x_detect: Raw detection position
            x_prev: Previous smoothed position
            x_t_minus_1: Position at t-1
            x_t_minus_2: Position at t-2
            
        Returns:
            Optimized position
        """
        # Simplified: Use Exponential Moving Average for efficiency
        # Full gradient descent terlalu berat untuk real-time
        
        x_smooth = (1 - self.alpha) * x_prev + self.alpha * x_detect
        
        # Optional: Gradient descent jika enable full optimization
        if self.iterations > 1:
            for _ in range(self.iterations):
                # External energy: align dengan detection (Blueprint Eq. 10)
                grad_external = self.w_external * (x_smooth - x_detect)
                
                # Temporal energy: minimize acceleration change (Blueprint Eq. 11)
                if len(x_t_minus_2) > 0:
                    v_current = x_smooth - x_t_minus_1
                    v_prev = x_t_minus_1 - x_t_minus_2
                    grad_temporal = self.k_temp * (v_current - v_prev)
                else:
                    grad_temporal = 0
                
                # Total gradient
                grad_total = grad_external + grad_temporal
                
                # Update dengan learning rate
                learning_rate = 0.15
                x_smooth = x_smooth - learning_rate * grad_total
        
        return x_smooth
    
    def _smooth_aspect_ratio(self, ar_detect: float, ar_prev: float) -> float:
        """
        Smooth aspect ratio (Blueprint Eq. 9)
        
        Args:
            ar_detect: Detected aspect ratio
            ar_prev: Previous smoothed aspect ratio
            
        Returns:
            Smoothed aspect ratio
        """
        # Internal energy: penalize AR change (Blueprint Eq. 9)
        ar_smooth = (1 - self.alpha * 0.5) * ar_prev + (self.alpha * 0.5) * ar_detect
        return ar_smooth
    
    def _compute_aspect_ratio(self, bbox: List[float]) -> float:
        """Compute aspect ratio dari bbox [x1, y1, x2, y2]"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width < 1e-6:
            return 1.0
        
        return height / width
    
    def _reconstruct_bbox(self, center: np.ndarray, original_bbox: List[float], 
                         ar_smooth: float) -> List[float]:
        """
        Reconstruct bbox dari smoothed center dan aspect ratio
        
        Args:
            center: Smoothed center [x, y]
            original_bbox: Original bbox [x1, y1, x2, y2]
            ar_smooth: Smoothed aspect ratio
            
        Returns:
            Reconstructed bbox [x1, y1, x2, y2]
        """
        # Original dimensions
        orig_width = original_bbox[2] - original_bbox[0]
        orig_height = original_bbox[3] - original_bbox[1]
        
        # Smooth dimensions preserving area
        area = orig_width * orig_height
        
        # h/w = ar → h = ar*w
        # area = w*h = w*(ar*w) = ar*w²
        # w = sqrt(area/ar)
        width = np.sqrt(area / max(ar_smooth, 0.5))
        height = ar_smooth * width
        
        # Reconstruct bbox
        x1 = center[0] - width / 2
        y1 = center[1] - height / 2
        x2 = center[0] + width / 2
        y2 = center[1] + height / 2
        
        return [int(x1), int(y1), int(x2), int(y2)]
    
    def smooth_batch(self, tracks: List) -> List:
        """
        Smooth detections untuk batch of tracks
        
        Args:
            tracks: List of track objects
            
        Returns:
            Tracks dengan smoothed detections
        """
        for track in tracks:
            if not hasattr(track, 'current_detection'):
                continue
            
            # Smooth current detection
            smoothed = self.smooth_detection(
                track.track_id, 
                track.current_detection
            )
            
            # Update track
            track.current_detection = smoothed
            
            # Update bbox and center references
            if hasattr(track, 'bbox'):
                track.bbox = smoothed['bbox']
            if hasattr(track, 'center'):
                track.center = smoothed['center']
        
        return tracks
    
    def reset_track(self, track_id: int):
        """Reset smoothed state untuk track tertentu"""
        if track_id in self.smoothed_states:
            del self.smoothed_states[track_id]
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove smoothed states untuk tracks yang sudah terminated"""
        terminated_ids = [tid for tid in self.smoothed_states.keys() 
                         if tid not in active_track_ids]
        
        for tid in terminated_ids:
            del self.smoothed_states[tid]
    
    def get_statistics(self) -> Dict:
        """Get EAGER smoother statistics"""
        return {
            'total_smooths': self.total_smooths,
            'active_tracks': len(self.smoothed_states),
            'alpha': self.alpha,
            'iterations': self.iterations,
            'config': self.config
        }
    
    def estimate_jitter_reduction(self, track_id: int, detection: Dict) -> float:
        """
        Estimate jitter reduction (raw vs smoothed displacement)
        
        Args:
            track_id: Track ID
            detection: Current detection
            
        Returns:
            Jitter reduction in pixels
        """
        if track_id not in self.smoothed_states:
            return 0.0
        
        state = self.smoothed_states[track_id]
        
        # Raw displacement
        raw_center = np.array(detection['center'])
        raw_displacement = np.linalg.norm(raw_center - state['prev_center'])
        
        # Smoothed displacement
        smooth_displacement = np.linalg.norm(state['x_smooth'] - state['prev_center'])
        
        # Reduction
        jitter_reduction = abs(raw_displacement - smooth_displacement)
        
        return jitter_reduction


# Testing
if __name__ == "__main__":
    print("Testing EAGER Smoother...")
    
    # Create smoother
    config = {
        'alpha': 0.2,
        'w_internal': 0.3,
        'w_external': 0.8,
        'k_temp': 0.5,
        'iterations': 5
    }
    
    smoother = EAGERSmoother(config)
    
    # Simulate jittery detections
    track_id = 1
    detections = [
        {'bbox': [100, 100, 150, 200], 'center': [125, 150]},
        {'bbox': [103, 98, 153, 198], 'center': [128, 148]},   # Jitter +3, -2
        {'bbox': [106, 102, 156, 202], 'center': [131, 152]},  # Jitter +3, +4
        {'bbox': [108, 99, 158, 199], 'center': [133, 149]},   # Jitter +2, -3
    ]
    
    print("\nSmoothing jittery sequence:")
    for i, det in enumerate(detections):
        smoothed = smoother.smooth_detection(track_id, det)
        
        print(f"Frame {i+1}:")
        print(f"  Raw center: {det['center']}")
        print(f"  Smoothed center: {smoothed['center']}")
        
        if i > 0:
            jitter_reduction = smoother.estimate_jitter_reduction(track_id, det)
            print(f"  Jitter reduction: {jitter_reduction:.2f} px")
    
    # Statistics
    stats = smoother.get_statistics()
    print(f"\n✓ EAGER Statistics:")
    print(f"  Total smooths: {stats['total_smooths']}")
    print(f"  Active tracks: {stats['active_tracks']}")
    
    print("\n✓ EAGER Smoother test completed")
