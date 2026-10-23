"""
Physics-Based Predictor for LTE-TrackGuard
===========================================

EXPERIMENTAL replacement untuk Kalman Filter menggunakan physics principles:
1. Momentum conservation (Blueprint Section 2.6.3)
2. Flow continuity via streamline simulation (Blueprint Section 2.6.4)
3. Energy signature matching (Blueprint Section 2.6.5)

USAGE NOTE:
This is experimental. Default mode tetap Kalman Filter.
Enable via PHYSICS_CONFIG['enable_physics_predictor'] = True untuk test.

From Blueprint Section 2.6: Physics-Based Occlusion Handling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class PhysicsPredictor:
    """
    Physics-based predictor sebagai alternative untuk Kalman Filter
    
    Uses conservation laws untuk predict track position during occlusion
    """
    
    def __init__(self, config: Dict):
        """
        Initialize physics predictor
        
        Args:
            config: Configuration dari PHYSICS_CONFIG['physics_predictor']
        """
        self.config = config
        
        # Parameters dari blueprint (Blueprint Section 2.6)
        self.sigma_momentum = config.get('sigma_momentum', 100.0)
        self.sigma_energy = config.get('sigma_energy', 500.0)
        
        # Cost matrix weights
        self.w_iou = config.get('w_iou', 0.35)
        self.w_momentum = config.get('w_momentum', 0.25)
        self.w_flow = config.get('w_flow', 0.25)
        self.w_energy = config.get('w_energy', 0.15)
        
        # Validation thresholds
        self.max_angle_diff = config.get('max_angle_diff', 45.0)  # degrees
        self.energy_ratio_min = config.get('energy_ratio_min', 0.5)
        self.energy_ratio_max = config.get('energy_ratio_max', 2.0)
        self.streamline_fit_max = config.get('streamline_fit_max', 50.0)  # pixels
        
        # Track state storage
        # Format: {track_id: {'momentum': array, 'energy': float, ...}}
        self.track_physics_states = {}
        
        # Statistics
        self.total_predictions = 0
        self.successful_reidentifications = 0
        self.physics_validations_passed = 0
        self.physics_validations_failed = 0
        
        print("✓ PhysicsPredictor initialized (EXPERIMENTAL)")
        print(f"  Momentum sigma: {self.sigma_momentum}")
        print(f"  Energy sigma: {self.sigma_energy}")
        print(f"  Cost weights: IoU={self.w_iou}, momentum={self.w_momentum}, "
              f"flow={self.w_flow}, energy={self.w_energy}")
        print("  ⚠️  This is experimental - may be less accurate than Kalman")
    
    def predict_track_position(self, track: object, velocity_field: 'VelocityField', 
                               frames_occluded: int = 1) -> np.ndarray:
        """
        Predict track position menggunakan physics (Blueprint Section 2.6.4)
        
        Uses streamline simulation instead of linear Kalman prediction
        
        Args:
            track: Track object
            velocity_field: VelocityField untuk streamline simulation
            frames_occluded: Number of frames in occlusion
            
        Returns:
            Predicted position [x, y]
        """
        self.total_predictions += 1
        
        # Get track's last known state
        if not hasattr(track, 'current_detection'):
            return np.array([0.0, 0.0])
        
        last_position = np.array(track.current_detection.get('center', [0, 0]))
        
        # Get track's velocity
        velocity = velocity_field.compute_velocity(track, dt=1.0)
        
        # Simple prediction: position + velocity * frames
        # (Streamline simulation requires global field from all tracks)
        predicted_position = last_position + velocity * frames_occluded
        
        return predicted_position
    
    def compute_physics_cost_matrix(self, ghost_tracks: List, detections: List[Dict],
                                    velocity_field: 'VelocityField', 
                                    iou_matrix: np.ndarray) -> np.ndarray:
        """
        Compute enhanced cost matrix dengan physics scores (Blueprint Section 2.6.6)
        
        C_physics[i,j] = w_iou * C_iou + w_momentum * (1-S_momentum) + 
                         w_flow * (1-S_flow) + w_energy * (1-S_energy)
        
        Args:
            ghost_tracks: List of ghost tracks
            detections: List of detections
            velocity_field: VelocityField
            iou_matrix: Standard IoU matrix
            
        Returns:
            Physics-enhanced cost matrix
        """
        n_tracks = len(ghost_tracks)
        n_detections = len(detections)
        
        if n_tracks == 0 or n_detections == 0:
            return np.zeros((n_tracks, n_detections))
        
        # Initialize cost matrix
        cost_matrix = np.zeros((n_tracks, n_detections))
        
        for i, track in enumerate(ghost_tracks):
            # Update physics state untuk track
            self._update_track_physics_state(track, velocity_field)
            
            track_state = self.track_physics_states.get(track.track_id, {})
            p_ghost = track_state.get('momentum', np.array([0.0, 0.0]))
            E_ghost = track_state.get('energy', 0.0)
            
            for j, detection in enumerate(detections):
                # IoU cost
                iou_cost = 1.0 - iou_matrix[i, j] if iou_matrix.size > 0 else 1.0
                
                # Momentum score (Blueprint Section 2.6.3)
                p_det = self._compute_momentum(detection, velocity_field)
                momentum_score = self._compute_momentum_score(p_ghost, p_det)
                
                # Flow continuity score (simplified - no global field here)
                flow_score = 0.5  # Default neutral score
                
                # Energy score (Blueprint Section 2.6.5)
                E_det = self._compute_energy(detection, velocity_field)
                energy_score = self._compute_energy_score(E_ghost, E_det)
                
                # Combined cost (Blueprint Eq. Section 2.6.6)
                cost_matrix[i, j] = (
                    self.w_iou * iou_cost +
                    self.w_momentum * (1 - momentum_score) +
                    self.w_flow * (1 - flow_score) +
                    self.w_energy * (1 - energy_score)
                )
        
        return cost_matrix
    
    def validate_physics_match(self, track: object, detection: Dict, 
                               velocity_field: 'VelocityField') -> bool:
        """
        Validate match menggunakan physics constraints (Blueprint Section 2.6.7)
        
        Args:
            track: Ghost track
            detection: Matched detection
            velocity_field: VelocityField
            
        Returns:
            True jika physics validation passed
        """
        # Get track physics state
        if track.track_id not in self.track_physics_states:
            return True  # No state to validate, allow match
        
        track_state = self.track_physics_states[track.track_id]
        p_ghost = track_state.get('momentum', np.array([0.0, 0.0]))
        E_ghost = track_state.get('energy', 0.0)
        
        # Detection physics
        p_det = self._compute_momentum(detection, velocity_field)
        E_det = self._compute_energy(detection, velocity_field)
        
        # Rule 1: Momentum direction consistency (Blueprint Section 2.6.7)
        if np.linalg.norm(p_ghost) > 1e-6 and np.linalg.norm(p_det) > 1e-6:
            cos_angle = np.dot(p_ghost, p_det) / (np.linalg.norm(p_ghost) * np.linalg.norm(p_det))
            angle_diff = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            
            if angle_diff > self.max_angle_diff:
                self.physics_validations_failed += 1
                return False  # Direction changed too much
        
        # Rule 2: Energy bound constraint (Blueprint Section 2.6.7)
        if E_ghost > 1e-6:
            energy_ratio = E_det / E_ghost
            
            if energy_ratio < self.energy_ratio_min or energy_ratio > self.energy_ratio_max:
                self.physics_validations_failed += 1
                return False  # Energy change too large
        
        # Passed all validations
        self.physics_validations_passed += 1
        return True
    
    def _update_track_physics_state(self, track: object, velocity_field: 'VelocityField'):
        """
        Update physics state untuk track (momentum, energy)
        
        Args:
            track: Track object
            velocity_field: VelocityField
        """
        # Compute momentum (Blueprint Section 2.6.3)
        velocity = velocity_field.compute_velocity(track, dt=1.0)
        bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
        mass = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # Proportional to area
        
        momentum = mass * velocity
        
        # Compute energy (Blueprint Section 2.6.5)
        energy = 0.5 * mass * np.linalg.norm(velocity)**2
        
        # Store state
        self.track_physics_states[track.track_id] = {
            'momentum': momentum,
            'energy': energy,
            'last_update_frame': track.current_frame
        }
    
    def _compute_momentum(self, detection: Dict, velocity_field: 'VelocityField') -> np.ndarray:
        """
        Compute momentum dari detection
        
        Args:
            detection: Detection dict
            velocity_field: VelocityField
            
        Returns:
            Momentum vector
        """
        # Simplified: assume velocity = 0 for new detection
        # In real usage, would need trajectory history
        bbox = detection.get('bbox', [0, 0, 0, 0])
        mass = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # No velocity info for detection, assume zero
        velocity = np.array([0.0, 0.0])
        
        return mass * velocity
    
    def _compute_energy(self, detection: Dict, velocity_field: 'VelocityField') -> float:
        """
        Compute kinetic energy dari detection
        
        Args:
            detection: Detection dict
            velocity_field: VelocityField
            
        Returns:
            Energy value
        """
        bbox = detection.get('bbox', [0, 0, 0, 0])
        mass = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # No velocity for detection
        velocity = np.array([0.0, 0.0])
        
        energy = 0.5 * mass * np.linalg.norm(velocity)**2
        
        return energy
    
    def _compute_momentum_score(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute momentum matching score (Blueprint Section 2.6.3)
        
        S_momentum = exp(-||p1 - p2||² / 2σ_p²)
        
        Args:
            p1: Momentum vector 1
            p2: Momentum vector 2
            
        Returns:
            Score [0, 1]
        """
        diff = np.linalg.norm(p1 - p2)
        score = np.exp(-diff**2 / (2 * self.sigma_momentum**2))
        
        return score
    
    def _compute_energy_score(self, E1: float, E2: float) -> float:
        """
        Compute energy matching score (Blueprint Section 2.6.5)
        
        S_energy = exp(-(E1 - E2)² / 2σ_E²)
        
        Args:
            E1: Energy 1
            E2: Energy 2
            
        Returns:
            Score [0, 1]
        """
        diff = abs(E1 - E2)
        score = np.exp(-diff**2 / (2 * self.sigma_energy**2))
        
        return score
    
    def cleanup_old_states(self, active_track_ids: List[int]):
        """
        Remove physics states untuk terminated tracks
        
        Args:
            active_track_ids: List of active track IDs
        """
        terminated_ids = [tid for tid in self.track_physics_states.keys() 
                         if tid not in active_track_ids]
        
        for tid in terminated_ids:
            del self.track_physics_states[tid]
    
    def get_statistics(self) -> Dict:
        """Get physics predictor statistics"""
        validation_rate = (self.physics_validations_passed / 
                          max(1, self.physics_validations_passed + self.physics_validations_failed))
        
        return {
            'total_predictions': self.total_predictions,
            'successful_reidentifications': self.successful_reidentifications,
            'physics_validations_passed': self.physics_validations_passed,
            'physics_validations_failed': self.physics_validations_failed,
            'validation_pass_rate': validation_rate,
            'active_physics_states': len(self.track_physics_states),
            'mode': 'experimental',
            'config': {
                'sigma_momentum': self.sigma_momentum,
                'sigma_energy': self.sigma_energy,
                'weights': {
                    'iou': self.w_iou,
                    'momentum': self.w_momentum,
                    'flow': self.w_flow,
                    'energy': self.w_energy
                }
            }
        }
    
    def compare_with_kalman(self, kalman_prediction: np.ndarray, 
                           physics_prediction: np.ndarray, 
                           ground_truth: np.ndarray) -> Dict:
        """
        Compare physics prediction vs Kalman prediction
        
        Args:
            kalman_prediction: Kalman filter prediction
            physics_prediction: Physics-based prediction
            ground_truth: Actual position
            
        Returns:
            Comparison metrics
        """
        kalman_error = np.linalg.norm(kalman_prediction - ground_truth)
        physics_error = np.linalg.norm(physics_prediction - ground_truth)
        
        return {
            'kalman_error': float(kalman_error),
            'physics_error': float(physics_error),
            'improvement': float(kalman_error - physics_error),
            'physics_better': physics_error < kalman_error
        }


# Testing
if __name__ == "__main__":
    print("Testing PhysicsPredictor...")
    
    # Create predictor
    config = {
        'sigma_momentum': 100.0,
        'sigma_energy': 500.0,
        'w_iou': 0.35,
        'w_momentum': 0.25,
        'w_flow': 0.25,
        'w_energy': 0.15,
        'max_angle_diff': 45.0,
        'energy_ratio_min': 0.5,
        'energy_ratio_max': 2.0,
        'streamline_fit_max': 50.0
    }
    
    predictor = PhysicsPredictor(config)
    
    # Create dummy track
    class DummyTrack:
        def __init__(self):
            self.track_id = 1
            self.current_frame = 10
            self.history = [
                {'center': [100, 100], 'bbox': [80, 85, 120, 115]},
                {'center': [110, 105], 'bbox': [90, 90, 130, 120]}
            ]
            self.current_detection = {
                'center': [120, 110],
                'bbox': [100, 95, 140, 125]
            }
    
    track = DummyTrack()
    
    # Create velocity field
    from physics.velocity_field import VelocityField
    vf_config = {'gaussian_radius': 100.0, 'flow_sigma': 50.0}
    vf = VelocityField(vf_config)
    
    print("\nTesting physics prediction:")
    
    # Test position prediction
    predicted_pos = predictor.predict_track_position(track, vf, frames_occluded=5)
    print(f"Predicted position (5 frames): {predicted_pos}")
    
    # Test momentum computation
    predictor._update_track_physics_state(track, vf)
    state = predictor.track_physics_states[track.track_id]
    print(f"Track momentum: {state['momentum']}")
    print(f"Track energy: {state['energy']:.2f}")
    
    # Test validation
    detection = {
        'center': [125, 112],
        'bbox': [105, 97, 145, 127]
    }
    
    is_valid = predictor.validate_physics_match(track, detection, vf)
    print(f"\nPhysics validation: {is_valid}")
    
    # Statistics
    stats = predictor.get_statistics()
    print(f"\n✓ PhysicsPredictor Statistics:")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Validation pass rate: {stats['validation_pass_rate']:.2%}")
    print(f"  Active physics states: {stats['active_physics_states']}")
    print(f"  Mode: {stats['mode']}")
    
    print("\n✓ PhysicsPredictor test completed")
    print("⚠️  NOTE: This is experimental - compare with Kalman before production use")
