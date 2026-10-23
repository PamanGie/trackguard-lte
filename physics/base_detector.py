"""
Base Detector for LTE-TrackGuard Behaviour Detection
====================================================

Abstract base class implementing state machine logic shared across all
behaviour detectors (turn, brake, fallen, collision).

STATE MACHINE:
- NORMAL: No anomaly detected
- MONITORING: Anomaly detected, observing for confirmation
- CONFIRMED: Persistent anomaly, trigger alert
- RECOVERY: Returning to normal state

From Blueprint Section 2.5: State Machine untuk Behaviour Confirmation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from collections import deque
import time


class BehaviourState:
    """State constants untuk behaviour detection"""
    NORMAL = 'normal'
    MONITORING = 'monitoring'
    CONFIRMED = 'confirmed'
    RECOVERY = 'recovery'


class BaseDetector(ABC):
    """
    Abstract base class untuk semua behaviour detectors
    
    Implements temporal persistence logic dan state machine dari Blueprint Section 2.5
    """
    
    def __init__(self, config: Dict, detector_name: str):
        """
        Initialize base detector
        
        Args:
            config: Configuration dictionary untuk detector
            detector_name: Nama detector (untuk logging)
        """
        self.detector_name = detector_name
        self.config = config
        
        # State machine parameters dari blueprint
        self.window_size = config.get('window_size', 5)
        self.persist_threshold = config.get('persist_threshold', 0.6)
        
        # Track-level state storage
        # Format: {track_id: {'state': str, 'onset_time': int, 'history': deque}}
        self.track_states = {}
        
        # Detection results storage
        self.detection_history = []
        
        # Performance tracking
        self.total_detections = 0
        self.confirmed_detections = 0
        self.false_positives = 0
        
        print(f"✓ {self.detector_name} initialized")
        print(f"  Window size: {self.window_size} frames")
        print(f"  Persist threshold: {self.persist_threshold}")
    
    @abstractmethod
    def _compute_metric(self, track: object, velocity_field: 'VelocityField') -> float:
        """
        Compute physics metric untuk detection (MUST be implemented by subclass)
        
        Args:
            track: Track object dengan current state
            velocity_field: VelocityField object untuk shared calculations
            
        Returns:
            Metric value (float)
        """
        pass
    
    @abstractmethod
    def _check_condition(self, metric_value: float, track: object) -> bool:
        """
        Check apakah metric value memenuhi condition untuk anomaly
        
        Args:
            metric_value: Computed metric value
            track: Track object
            
        Returns:
            True jika condition terpenuhi (anomaly detected)
        """
        pass
    
    def detect(self, tracks: List, velocity_field: 'VelocityField') -> List[Dict]:
        """
        Main detection method - process all tracks
        
        Args:
            tracks: List of track objects
            velocity_field: VelocityField untuk physics calculations
            
        Returns:
            List of confirmed detections
        """
        detections = []
        
        for track in tracks:
            # Skip tracks yang tidak valid
            if not self._is_valid_track(track):
                continue
            
            # Compute physics metric
            metric_value = self._compute_metric(track, velocity_field)
            
            # Check condition
            condition_met = self._check_condition(metric_value, track)
            
            # NOTE: Bypass untuk FallenDetector dihapus karena sudah di-handle di FallenDetector.detect()
            # FallenDetector memiliki logic sendiri untuk bypass di method detect() yang override
            
            # Update state machine
            detection_result = self._update_state_machine(
                track.track_id, 
                condition_met, 
                metric_value,
                track
            )
            
            # Collect confirmed detections
            if detection_result is not None:
                detections.append(detection_result)
        
        # Cleanup terminated tracks
        self._cleanup_old_states(tracks)
        
        return detections
    
    def _update_state_machine(self, track_id: int, condition_met: bool, 
                             metric_value: float, track: object) -> Optional[Dict]:
        """
        Update state machine untuk track (Blueprint Section 2.5)
        
        Returns:
            Detection dict jika CONFIRMED, None otherwise
        """
        # Initialize state untuk track baru
        if track_id not in self.track_states:
            self.track_states[track_id] = {
                'state': BehaviourState.NORMAL,
                'onset_time': 0,
                'history': deque(maxlen=self.window_size),
                'metric_history': deque(maxlen=self.window_size)
            }
        
        state_info = self.track_states[track_id]
        current_state = state_info['state']
        
        # Update history
        state_info['history'].append(1 if condition_met else 0)
        state_info['metric_history'].append(metric_value)
        
        # Calculate persistence
        persistence = self._calculate_persistence(state_info['history'])
        
        # State transitions (Blueprint Section 2.5.2)
        if current_state == BehaviourState.NORMAL:
            if condition_met:
                state_info['state'] = BehaviourState.MONITORING
                state_info['onset_time'] = 1
                
        elif current_state == BehaviourState.MONITORING:
            state_info['onset_time'] += 1
            
            # SIMPLIFIED: Kurangi requirement untuk video pendek
            if persistence > self.persist_threshold and state_info['onset_time'] >= 1:
                # Transition to CONFIRMED
                state_info['state'] = BehaviourState.CONFIRMED
                self.confirmed_detections += 1
                
                # Create detection result
                return self._create_detection_result(track, metric_value, persistence)
            
            elif not condition_met:
                # False alarm, back to NORMAL
                state_info['state'] = BehaviourState.NORMAL
                state_info['onset_time'] = 0
                
        elif current_state == BehaviourState.CONFIRMED:
            if not condition_met:
                # Start recovery
                state_info['state'] = BehaviourState.RECOVERY
                state_info['onset_time'] = 0
                
        elif current_state == BehaviourState.RECOVERY:
            state_info['onset_time'] += 1
            
            if state_info['onset_time'] >= 5:
                # Stabil, back to NORMAL
                state_info['state'] = BehaviourState.NORMAL
                state_info['onset_time'] = 0
        
        return None
    
    def _calculate_persistence(self, history: deque) -> float:
        """
        Calculate temporal persistence metric (Blueprint Eq. untuk P_b,i)
        
        Args:
            history: Deque of binary indicators (1=condition met, 0=not met)
            
        Returns:
            Persistence score [0, 1]
        """
        if len(history) == 0:
            return 0.0
        
        return sum(history) / len(history)
    
    def _create_detection_result(self, track: object, metric_value: float, 
                                 persistence: float) -> Dict:
        """
        Create detection result dictionary
        
        Args:
            track: Track object
            metric_value: Physics metric value
            persistence: Persistence score
            
        Returns:
            Detection result dict
        """
        bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
        center = track.current_detection.get('center', [0, 0])
        
        return {
            'track_id': track.track_id,
            'frame_id': track.current_frame,
            'detector': self.detector_name,
            'metric_value': float(metric_value),
            'persistence': float(persistence),
            'bbox': bbox,
            'center': center,
            'timestamp': time.time(),
            'class_name': track.current_detection.get('class_name', 'unknown')
        }
    
    def _is_valid_track(self, track: object) -> bool:
        """
        Check apakah track valid untuk detection
        
        Args:
            track: Track object
            
        Returns:
            True jika valid
        """
        # Skip track yang baru dibuat (hits < 3)
        if not hasattr(track, 'hits') or track.hits < 3:
            return False
        
        # Skip ghost tracks
        if hasattr(track, 'state') and track.state == 'ghost':
            return False
        
        # Must have current detection
        if not hasattr(track, 'current_detection'):
            return False
        
        return True
    
    def _cleanup_old_states(self, tracks: List):
        """
        Remove state info untuk tracks yang sudah terminated
        
        Args:
            tracks: List of active tracks
        """
        active_track_ids = {track.track_id for track in tracks}
        terminated_ids = [tid for tid in self.track_states.keys() 
                         if tid not in active_track_ids]
        
        for tid in terminated_ids:
            del self.track_states[tid]
    
    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        return {
            'detector_name': self.detector_name,
            'total_detections': self.total_detections,
            'confirmed_detections': self.confirmed_detections,
            'false_positives': self.false_positives,
            'active_states': len(self.track_states),
            'config': self.config
        }
    
    def reset(self):
        """Reset detector state"""
        self.track_states.clear()
        self.detection_history.clear()
        self.total_detections = 0
        self.confirmed_detections = 0
        self.false_positives = 0


# Testing
if __name__ == "__main__":
    print("Testing BaseDetector state machine...")
    
    # Create dummy detector implementation
    class DummyDetector(BaseDetector):
        def __init__(self):
            config = {'window_size': 5, 'persist_threshold': 0.6}
            super().__init__(config, "DummyDetector")
        
        def _compute_metric(self, track, velocity_field):
            return np.random.random()  # Random metric
        
        def _check_condition(self, metric_value, track):
            return metric_value > 0.7  # Threshold
    
    detector = DummyDetector()
    print(f"✓ {detector.detector_name} created")
    print(f"  State machine ready: {len(detector.track_states)} active states")
    print("✓ BaseDetector test completed")
