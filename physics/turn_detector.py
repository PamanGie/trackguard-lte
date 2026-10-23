"""
Turn Detector for LTE-TrackGuard
=================================

Deteksi belok kiri/kanan menggunakan vorticity (cross product 2D).

CP_i(t) = v_x,i(t-1) · v_y,i(t) - v_y,i(t-1) · v_x,i(t)

- CP > 0: Belok kanan (clockwise)
- CP < 0: Belok kiri (counter-clockwise)
- CP ≈ 0: Lurus (laminar flow)

From Blueprint Section 2.4.2: Behaviour 1 - Deteksi Belok Kiri/Kanan
"""

import numpy as np
from typing import Dict, List, Optional
from physics.base_detector import BaseDetector, BehaviourState


class TurnDetector(BaseDetector):
    """
    Detector untuk belok kiri/kanan menggunakan vorticity
    
    Physics metric: Cross Product 2D dari velocity vectors
    """
    
    def __init__(self, config: Dict):
        """
        Initialize turn detector
        
        Args:
            config: Configuration dari PHYSICS_CONFIG['turn_detector']
        """
        super().__init__(config, "TurnDetector")
        
        # Thresholds dari blueprint (Blueprint Section 2.4.2)
        self.tau_turn_right = config.get('tau_turn_right', 0.5)   # CP > 0.5
        self.tau_turn_left = config.get('tau_turn_left', -0.5)    # CP < -0.5
        self.tau_neutral = config.get('tau_neutral', 0.15)        # |CP| < 0.15
        
        # Zig-zag detection (aggressive driving)
        self.detect_zigzag = config.get('detect_zigzag', True)
        self.zigzag_window = 10  # frames
        self.zigzag_threshold = 0.6  # 60% sign changes
        
        print(f"  Turn right threshold: {self.tau_turn_right}")
        print(f"  Turn left threshold: {self.tau_turn_left}")
        print(f"  Neutral threshold: {self.tau_neutral}")
        print(f"  Detect zig-zag: {self.detect_zigzag}")
    
    def _compute_metric(self, track: object, velocity_field: 'VelocityField') -> float:
        """
        Compute cross product untuk turn detection
        
        Args:
            track: Track object
            velocity_field: VelocityField untuk cross product calculation
            
        Returns:
            Cross product value (+ = right, - = left, 0 = straight)
        """
        # Use velocity_field's built-in cross product calculator
        cross_product = velocity_field.compute_cross_product_2d(track)
        
        return cross_product
    
    def _check_condition(self, metric_value: float, track: object) -> bool:
        """
        Check apakah turn condition terpenuhi
        
        Args:
            metric_value: Cross product value
            track: Track object
            
        Returns:
            True jika turning detected (either left or right)
        """
        # Turn right OR turn left
        is_turning = (metric_value > self.tau_turn_right or 
                     metric_value < self.tau_turn_left)
        
        return is_turning
    
    def detect(self, tracks: List, velocity_field: 'VelocityField') -> List[Dict]:
        """
        Override detect untuk add zig-zag detection
        
        Args:
            tracks: List of track objects
            velocity_field: VelocityField object
            
        Returns:
            List of confirmed turn detections (including zig-zag)
        """
        # Filter: HANYA process motorcycle, skip person atau class lain
        motorcycle_tracks = []
        for track in tracks:
            if not self._is_valid_track(track):
                continue
            class_name = track.current_detection.get('class_name', 'unknown')
            if class_name == 'motorcycle':
                motorcycle_tracks.append(track)
        
        # Normal turn detection (hanya untuk motorcycle)
        detections = super().detect(motorcycle_tracks, velocity_field)
        
        # Additional zig-zag detection
        if self.detect_zigzag:
            zigzag_detections = self._detect_zigzag_pattern(tracks, velocity_field)
            detections.extend(zigzag_detections)
        
        return detections
    
    def _detect_zigzag_pattern(self, tracks: List, velocity_field: 'VelocityField') -> List[Dict]:
        """
        Detect zig-zag / aggressive driving pattern (Blueprint Section 2.4.2)
        
        Zig-zag Score = (1/W) Σ |sign(CP(k)) - sign(CP(k-1))|
        
        Args:
            tracks: List of tracks
            velocity_field: VelocityField
            
        Returns:
            List of zig-zag detections
        """
        zigzag_detections = []
        
        for track in tracks:
            if not self._is_valid_track(track):
                continue
            
            # Initialize state untuk track baru
            if track.track_id not in self.track_states:
                continue
            
            state_info = self.track_states[track.track_id]
            
            # Need enough history for zig-zag
            if len(state_info['metric_history']) < self.zigzag_window:
                continue
            
            # Get recent cross product history
            cp_history = list(state_info['metric_history'])[-self.zigzag_window:]
            
            # Count sign changes
            sign_changes = 0
            for i in range(1, len(cp_history)):
                cp_prev = cp_history[i-1]
                cp_curr = cp_history[i]
                
                # Check sign change (skip neutral values)
                if abs(cp_prev) > self.tau_neutral and abs(cp_curr) > self.tau_neutral:
                    if np.sign(cp_prev) != np.sign(cp_curr):
                        sign_changes += 1
            
            # Zig-zag score
            zigzag_score = sign_changes / (len(cp_history) - 1) if len(cp_history) > 1 else 0
            
            # Check threshold
            if zigzag_score > self.zigzag_threshold:
                # Create zig-zag detection
                detection = self._create_zigzag_detection(track, zigzag_score)
                zigzag_detections.append(detection)
        
        return zigzag_detections
    
    def _create_detection_result(self, track: object, metric_value: float, 
                                 persistence: float) -> Dict:
        """
        Create turn detection result dengan direction info
        
        Args:
            track: Track object
            metric_value: Cross product value
            persistence: Persistence score
            
        Returns:
            Detection result dict
        """
        result = super()._create_detection_result(track, metric_value, persistence)
        
        # Determine turn direction
        if metric_value > self.tau_turn_right:
            direction = 'right'
            direction_id = 1
        elif metric_value < self.tau_turn_left:
            direction = 'left'
            direction_id = -1
        else:
            direction = 'straight'
            direction_id = 0
        
        # Add turn-specific info
        result.update({
            'behaviour_type': 'turn',
            'turn_direction': direction,
            'turn_direction_id': direction_id,
            'cross_product': float(metric_value),
            'severity': 'low',  # Normal turn
            'alert_level': 'info'
        })
        
        return result
    
    def _create_zigzag_detection(self, track: object, zigzag_score: float) -> Dict:
        """
        Create zig-zag detection result
        
        Args:
            track: Track object
            zigzag_score: Zig-zag score [0, 1]
            
        Returns:
            Detection dict
        """
        bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
        center = track.current_detection.get('center', [0, 0])
        
        return {
            'track_id': track.track_id,
            'frame_id': track.current_frame,
            'detector': self.detector_name,
            'behaviour_type': 'zigzag_driving',
            'metric_value': float(zigzag_score),
            'persistence': 1.0,  # Zig-zag based on window analysis
            'bbox': bbox,
            'center': center,
            'class_name': track.current_detection.get('class_name', 'unknown'),
            'severity': 'medium',  # Aggressive driving
            'alert_level': 'warning'
        }


# Testing
if __name__ == "__main__":
    print("Testing TurnDetector...")
    
    # Create detector
    config = {
        'tau_turn_right': 0.5,
        'tau_turn_left': -0.5,
        'tau_neutral': 0.15,
        'window_size': 5,
        'persist_threshold': 0.6,
        'detect_zigzag': True
    }
    
    detector = TurnDetector(config)
    
    # Create dummy track - car turning right
    class DummyTrack:
        def __init__(self, centers):
            self.track_id = 1
            self.state = 'active'
            self.hits = 10
            self.current_frame = len(centers)
            
            self.history = [{'center': c, 'bbox': [c[0]-20, c[1]-15, c[0]+20, c[1]+15]} 
                           for c in centers[:-1]]
            self.current_detection = {
                'center': centers[-1],
                'bbox': [centers[-1][0]-20, centers[-1][1]-15, 
                        centers[-1][0]+20, centers[-1][1]+15],
                'class_name': 'car'
            }
    
    # Straight trajectory
    straight_track = DummyTrack([[100+i*5, 100] for i in range(8)])
    
    # Right turn trajectory (curving right = positive CP)
    right_turn_track = DummyTrack([
        [100, 100], [105, 100], [110, 101], [115, 103], 
        [120, 106], [125, 110], [130, 115], [135, 121]
    ])
    
    # Left turn trajectory
    left_turn_track = DummyTrack([
        [100, 100], [105, 100], [110, 99], [115, 97], 
        [120, 94], [125, 90], [130, 85], [135, 79]
    ])
    
    # Create velocity field
    from physics.velocity_field import VelocityField
    vf_config = {'gaussian_radius': 100.0, 'flow_sigma': 50.0}
    vf = VelocityField(vf_config)
    
    print("\nTesting turn detection:")
    
    # Test straight
    cp_straight = detector._compute_metric(straight_track, vf)
    print(f"\nStraight track CP: {cp_straight:.3f} (should be ~0)")
    
    # Test right turn
    cp_right = detector._compute_metric(right_turn_track, vf)
    print(f"Right turn track CP: {cp_right:.3f} (should be > 0.5)")
    is_right = detector._check_condition(cp_right, right_turn_track)
    print(f"  Right turn detected: {is_right}")
    
    # Test left turn
    cp_left = detector._compute_metric(left_turn_track, vf)
    print(f"Left turn track CP: {cp_left:.3f} (should be < -0.5)")
    is_left = detector._check_condition(cp_left, left_turn_track)
    print(f"  Left turn detected: {is_left}")
    
    # Full detection
    tracks = [straight_track, right_turn_track, left_turn_track]
    detections = detector.detect(tracks, vf)
    
    print(f"\n✓ Total detections: {len(detections)}")
    for det in detections:
        print(f"  Track {det['track_id']}: {det['behaviour_type']} "
              f"{det.get('turn_direction', 'N/A')} (CP={det['metric_value']:.3f})")
    
    # Statistics
    stats = detector.get_statistics()
    print(f"\n✓ TurnDetector Statistics:")
    print(f"  Confirmed detections: {stats['confirmed_detections']}")
    
    print("\n✓ TurnDetector test completed")
