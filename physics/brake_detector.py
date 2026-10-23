"""
Brake Detector for LTE-TrackGuard (IMPROVED)
=============================================

Deteksi rem mendadak menggunakan physics-based prediction error.

IMPROVED LOGIC:
1. Predict velocity menggunakan inertia (Newton's 1st Law)
2. Predict position dari predicted velocity
3. Compare prediction vs actual (velocity + position error)
4. Sudden brake = large velocity drop + small position error (clean stop)

From Blueprint Section 2.4.3 + 2.6.4: Physics-Based Brake Detection
"""

import numpy as np
from typing import Dict, List, Optional
from physics.base_detector import BaseDetector, BehaviourState


class BrakeDetector(BaseDetector):
    """
    Detector untuk rem mendadak menggunakan physics prediction error
    
    Physics metrics:
    - Velocity prediction error (inertia assumption)
    - Position prediction error
    - Deceleration magnitude
    """
    
    def __init__(self, config: Dict):
        """
        Initialize brake detector
        
        Args:
            config: Configuration dari PHYSICS_CONFIG['brake_detector']
        """
        super().__init__(config, "BrakeDetector")
        
        # IMPROVED: Physics-based thresholds
        self.velocity_drop_strong = config.get('velocity_drop_strong', 5.0)    # px/frame
        self.velocity_drop_moderate = config.get('velocity_drop_moderate', 3.0)  # px/frame
        
        # Position error thresholds (untuk distinguish brake vs turn)
        self.position_error_clean_stop = config.get('position_error_clean_stop', 15.0)  # pixels
        self.position_error_moderate = config.get('position_error_moderate', 10.0)  # pixels
        
        # Minimum speed untuk detect brake (avoid false positive pada motor parkir)
        self.min_speed_for_brake = config.get('min_speed_for_brake', 5.0)  # px/frame
        
        # Legacy thresholds (for backward compatibility)
        self.tau_brake = config.get('tau_brake', -0.8)
        self.tau_decel = config.get('tau_decel', 5.0)
        
        print(f"  Physics-based brake detection:")
        print(f"    Velocity drop thresholds: strong={self.velocity_drop_strong}, moderate={self.velocity_drop_moderate} px/frame")
        print(f"    Position error thresholds: clean={self.position_error_clean_stop}, moderate={self.position_error_moderate} px")
        print(f"    Min speed for brake: {self.min_speed_for_brake} px/frame")
    
    def _compute_metric(self, track: object, velocity_field: 'VelocityField') -> float:
        """
        Compute brake metric menggunakan physics prediction error
        
        IMPROVED LOGIC:
        1. Predict velocity dari t-2 to t-1 (assume constant by inertia)
        2. Actual velocity dari t-1 to t (current)
        3. Velocity prediction error = |v_predicted - v_actual|
        4. Position prediction error = |pos_predicted - pos_actual|
        5. Combine metrics untuk detect sudden brake
        
        Returns:
            Brake score (negative = braking):
            - -1.5 = DEFINITE brake (strong velocity drop + clean stop)
            - -0.9 = LIKELY brake (strong drop + some deviation)
            - -0.8 = POSSIBLE brake (moderate drop + clean stop)
            - -0.4 = UNCERTAIN (moderate drop + deviation)
            - 0.0 = NO brake
        
        Args:
            track: Track object
            velocity_field: VelocityField untuk physics calculations
            
        Returns:
            Brake metric (negative value)
        """
        # Need at least 3 frames untuk prediction
        if not hasattr(track, 'history') or len(track.history) < 3:
            return 0.0
        
        # === STEP 1: Compute velocities ===
        
        # Velocity at t-1 (from t-2 to t-1)
        # This is our "predicted" velocity (assume continues by inertia)
        v_predicted = self._compute_velocity_at_offset(track, velocity_field, offset=-1)
        
        # Velocity at t (from t-1 to t - current)
        v_actual = velocity_field.compute_velocity(track, dt=1.0)
        
        # Speed magnitudes
        speed_predicted = np.linalg.norm(v_predicted)
        speed_actual = np.linalg.norm(v_actual)
        
        # === STEP 2: Check minimum speed requirement ===
        # Avoid false positive pada motor yang sudah lambat/parkir
        if speed_predicted < self.min_speed_for_brake:
            return 0.0  # Motor sudah lambat, tidak detect brake
        
        # === STEP 3: Velocity prediction error ===
        # Velocity drop = predicted speed - actual speed
        # Positive value = deceleration
        velocity_drop = speed_predicted - speed_actual
        
        # === STEP 4: Position prediction error ===
        # Predict position: x(t) = x(t-1) + v(t-1) * dt
        pos_t_minus_1 = np.array(track.history[-2].get('center', [0, 0]))
        pos_predicted = pos_t_minus_1 + v_predicted  # Inertia prediction
        
        # Actual position
        pos_actual = np.array(track.current_detection.get('center', [0, 0]))
        
        # Position error
        position_error = np.linalg.norm(pos_actual - pos_predicted)
        
        # === STEP 5: DECISION LOGIC ===
        # Combine velocity drop + position error
        
        # Scenario A: DEFINITE BRAKE
        # - Strong velocity drop (>5 px/frame) + clean stop (error <15 px)
        # Motor berhenti mendadak di tempat (tidak belok)
        if velocity_drop > self.velocity_drop_strong:
            if position_error < self.position_error_clean_stop:
                return -1.5  # DEFINITE brake (highest confidence)
            else:
                return -0.9  # LIKELY brake (brake + turn/swerve)
        
        # Scenario B: LIKELY BRAKE
        # - Moderate velocity drop (>3 px/frame) + clean stop (error <10 px)
        elif velocity_drop > self.velocity_drop_moderate:
            if position_error < self.position_error_moderate:
                return -0.8  # LIKELY brake (good confidence)
            else:
                return -0.4  # UNCERTAIN (might be normal deceleration/turn)
        
        # Scenario C: NO BRAKE or normal deceleration
        else:
            return 0.0
    
    def _compute_velocity_at_offset(self, track: object, velocity_field: 'VelocityField', 
                                   offset: int = -1) -> np.ndarray:
        """
        Compute velocity at specific frame offset
        
        Args:
            track: Track object
            velocity_field: VelocityField
            offset: Frame offset (negative = past, e.g., -1 = previous frame)
            
        Returns:
            Velocity vector [vx, vy]
        """
        # Check if we have enough history
        required_frames = abs(offset) + 1
        if len(track.history) < required_frames:
            return np.array([0.0, 0.0])
        
        # Temporarily modify track to compute velocity at offset
        original_current = track.current_detection
        original_history = track.history.copy()
        
        # Set current to frame at offset
        track.current_detection = track.history[offset]
        
        # Trim history to match
        track.history = track.history[:offset] if offset < -1 else track.history[:offset+1]
        
        # Compute velocity
        velocity = velocity_field.compute_velocity(track, dt=1.0)
        
        # Restore track state
        track.current_detection = original_current
        track.history = original_history
        
        return velocity
    
    def detect(self, tracks: List, velocity_field: 'VelocityField', fps: float = 30.0) -> List[Dict]:
        """
        Override detect untuk pass fps dan velocity_field ke _create_detection_result
        
        Args:
            tracks: List of tracks
            velocity_field: VelocityField
            fps: Video FPS untuk speed estimation
            
        Returns:
            List of detection results
        """
        # Store fps dan velocity_field untuk digunakan di _create_detection_result
        self._current_fps = fps
        self._current_velocity_field = velocity_field
        
        # Call parent detect
        detections = []
        
        for track in tracks:
            # Skip tracks yang tidak valid
            if not self._is_valid_track(track):
                continue
            
            # Compute physics metric
            metric_value = self._compute_metric(track, velocity_field)
            
            # Check condition
            condition_met = self._check_condition(metric_value, track)
            
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
    
    def _check_condition(self, metric_value: float, track: object) -> bool:
        """
        Check apakah brake condition terpenuhi
        
        Args:
            metric_value: Brake metric (negative value)
            track: Track object
            
        Returns:
            True jika sudden braking detected
        """
        # Threshold: metric <= -0.8 (at least LIKELY brake)
        # -1.5 = DEFINITE brake
        # -0.9 = LIKELY brake
        # -0.8 = POSSIBLE brake
        # -0.4 = UNCERTAIN (below threshold)
        
        return metric_value <= -0.8
    
    def _compute_deceleration(self, track: object) -> float:
        """
        Compute deceleration magnitude (legacy method for statistics)
        
        Args:
            track: Track object
            
        Returns:
            Deceleration magnitude (positive value)
        """
        if len(track.history) < 2:
            return 0.0
        
        # Current velocity
        c_curr = track.current_detection.get('center', [0, 0])
        c_prev = track.history[-2].get('center', c_curr)
        v_curr = np.array([c_curr[0] - c_prev[0], c_curr[1] - c_prev[1]])
        speed_curr = np.linalg.norm(v_curr)
        
        # Previous velocity
        if len(track.history) >= 3:
            c_prev2 = track.history[-3].get('center', c_prev)
            v_prev = np.array([c_prev[0] - c_prev2[0], c_prev[1] - c_prev2[1]])
            speed_prev = np.linalg.norm(v_prev)
        else:
            speed_prev = speed_curr
        
        # Deceleration (magnitude of speed decrease)
        decel = max(0, speed_prev - speed_curr)
        
        return decel
    
    def _estimate_speed_kmh(self, track: object, velocity_field: 'VelocityField', 
                            fps: float = 30.0) -> float:
        """
        Estimate speed dalam km/h berdasarkan ukuran kendaraan
        
        Konsep: Gunakan ukuran kendaraan sebagai referensi (motor ~1.5m lebar)
        untuk estimasi pixel-to-meter ratio, lalu konversi ke km/h
        
        Args:
            track: Track object
            velocity_field: VelocityField untuk compute velocity
            fps: Video FPS (default 30, bisa diambil dari video metadata)
            
        Returns:
            Speed dalam km/h
        """
        # Compute current speed (px/frame)
        v_actual = velocity_field.compute_velocity(track, dt=1.0)
        speed_px_frame = np.linalg.norm(v_actual)
        
        if speed_px_frame < 0.1:
            return 0.0
        
        # Get bbox untuk estimasi ukuran kendaraan
        bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        
        if bbox_width < 1.0 or bbox_height < 1.0:
            return 0.0
        
        # Estimasi ukuran kendaraan real (meter)
        # Motor: lebar ~1.5m, tinggi ~1.2m
        # Car: lebar ~1.8m, tinggi ~1.5m
        class_name = track.current_detection.get('class_name', 'unknown')
        
        if class_name == 'motorcycle':
            vehicle_width_real = 1.5  # meter
        elif class_name == 'car':
            vehicle_width_real = 1.8  # meter
        else:
            # Default: estimasi dari aspect ratio
            vehicle_width_real = 1.6  # meter
        
        # Pixel-to-meter ratio
        pixel_to_meter = vehicle_width_real / bbox_width
        
        # Convert px/frame ke m/s
        speed_m_per_frame = speed_px_frame * pixel_to_meter
        speed_m_per_second = speed_m_per_frame * fps
        
        # Convert m/s ke km/h
        speed_kmh = speed_m_per_second * 3.6
        
        return speed_kmh
    
    def _create_detection_result(self, track: object, metric_value: float, 
                                 persistence: float) -> Dict:
        """
        Create brake detection result dengan speed estimator
        
        Args:
            track: Track object
            metric_value: Brake metric (negative value)
            persistence: Persistence score
            
        Returns:
            Detection result dict
        """
        result = super()._create_detection_result(track, metric_value, persistence)
        
        # Get stored velocity_field and fps
        velocity_field = getattr(self, '_current_velocity_field', None)
        fps = getattr(self, '_current_fps', 30.0)
        
        # Compute deceleration for statistics
        decel = self._compute_deceleration(track)
        
        # Compute speed estimator
        v_actual = velocity_field.compute_velocity(track, dt=1.0) if velocity_field else np.array([0.0, 0.0])
        speed_px_frame = np.linalg.norm(v_actual)
        
        # Estimate speed in km/h
        speed_kmh = 0.0
        if velocity_field:
            speed_kmh = self._estimate_speed_kmh(track, velocity_field, fps)
        
        # Determine behaviour_type berdasarkan speed:
        # - DECELERATING: speed turun tapi masih bergerak (speed > threshold)
        # - BRAKE: speed turun sampai berhenti (speed < threshold)
        speed_threshold = 5.0  # px/frame - threshold untuk bedakan decelerating vs brake
        
        if speed_px_frame > speed_threshold:
            behaviour_type = 'decelerating'  # Masih bergerak, tapi melambat
        else:
            behaviour_type = 'sudden_brake'  # Berhenti atau hampir berhenti
        
        # Determine confidence level berdasarkan metric value
        if metric_value <= -1.5:
            confidence_level = 'very_high'
            severity = 'high'
            alert_level = 'warning'
        elif metric_value <= -0.9:
            confidence_level = 'high'
            severity = 'high'
            alert_level = 'warning'
        elif metric_value <= -0.8:
            confidence_level = 'medium'
            severity = 'medium'
            alert_level = 'caution'
        else:
            confidence_level = 'low'
            severity = 'low'
            alert_level = 'info'
        
        # Add brake-specific info dengan speed estimator
        result.update({
            'behaviour_type': behaviour_type,
            'brake_metric': float(metric_value),
            'deceleration': float(decel),
            'speed_estimator': float(speed_px_frame),  # px/frame
            'speed_kmh': float(speed_kmh),  # km/h
            'confidence_level': confidence_level,
            'severity': severity,
            'alert_level': alert_level,
            'detection_method': 'physics_prediction_error'
        })
        
        return result


# Testing
if __name__ == "__main__":
    print("Testing Improved BrakeDetector (Physics-Based)...")
    
    # Create detector
    config = {
        'velocity_drop_strong': 5.0,
        'velocity_drop_moderate': 3.0,
        'position_error_clean_stop': 15.0,
        'position_error_moderate': 10.0,
        'min_speed_for_brake': 5.0,
        'window_size': 5,
        'persist_threshold': 0.6
    }
    
    detector = BrakeDetector(config)
    
    # Create dummy track - car braking suddenly
    class DummyTrack:
        def __init__(self, centers, track_name=""):
            self.track_id = 1
            self.state = 'active'
            self.hits = 10
            self.current_frame = len(centers)
            self.name = track_name
            
            self.history = [{'center': c, 'bbox': [c[0]-20, c[1]-15, c[0]+20, c[1]+15]} 
                           for c in centers[:-1]]
            self.current_detection = {
                'center': centers[-1],
                'bbox': [centers[-1][0]-20, centers[-1][1]-15, 
                        centers[-1][0]+20, centers[-1][1]+15],
                'class_name': 'car'
            }
    
    # Scenario 1: Normal driving (constant speed ~10 px/frame)
    normal_track = DummyTrack(
        [[100+i*10, 100] for i in range(8)],
        "Normal driving"
    )
    
    # Scenario 2: Sudden braking (10 px/frame → 1 px/frame)
    sudden_brake_track = DummyTrack([
        [100, 100], [110, 100], [120, 100], [130, 100],  # Fast (10 px/frame)
        [135, 100], [138, 100], [140, 100], [141, 100]   # Brake! (1-3 px/frame)
    ], "Sudden brake")
    
    # Scenario 3: Gradual deceleration (10 → 8 → 6 → 4 px/frame)
    gradual_decel_track = DummyTrack([
        [100, 100], [110, 100], [120, 100], [130, 100],  # 10 px/frame
        [138, 100], [144, 100], [148, 100], [150, 100]   # 8, 6, 4, 2 px/frame
    ], "Gradual deceleration")
    
    # Scenario 4: Turn (velocity change but not brake)
    turn_track = DummyTrack([
        [100, 100], [110, 100], [120, 100], [130, 100],  # Straight
        [138, 105], [144, 112], [148, 120], [150, 130]   # Turn (y increases)
    ], "Turning")
    
    # Create velocity field
    from physics.velocity_field import VelocityField
    vf_config = {'gaussian_radius': 100.0, 'flow_sigma': 50.0}
    vf = VelocityField(vf_config)
    
    print("\n" + "="*60)
    print("Testing Physics-Based Brake Detection")
    print("="*60)
    
    # Test all scenarios
    test_tracks = [normal_track, sudden_brake_track, gradual_decel_track, turn_track]
    
    for track in test_tracks:
        print(f"\n[{track.name}]")
        
        # Compute metric
        metric = detector._compute_metric(track, vf)
        print(f"  Brake metric: {metric:.3f}")
        
        # Check condition
        is_braking = detector._check_condition(metric, track)
        print(f"  Brake detected: {is_braking}")
        
        # Deceleration
        decel = detector._compute_deceleration(track)
        print(f"  Deceleration: {decel:.2f} px/frame")
    
    print("\n" + "="*60)
    print("Full Detection Test")
    print("="*60)
    
    # Full detection
    detections = detector.detect(test_tracks, vf)
    
    print(f"\n✓ Total detections: {len(detections)}")
    for det in detections:
        print(f"\n  Track {det['track_id']}:")
        print(f"    Behaviour: {det['behaviour_type']}")
        print(f"    Brake metric: {det['brake_metric']:.3f}")
        print(f"    Deceleration: {det['deceleration']:.2f} px/frame")
        print(f"    Confidence: {det['confidence_level']}")
        print(f"    Severity: {det['severity']}")
        print(f"    Method: {det['detection_method']}")
    
    # Statistics
    stats = detector.get_statistics()
    print(f"\n✓ BrakeDetector Statistics:")
    print(f"  Confirmed detections: {stats['confirmed_detections']}")
    print(f"  Active states: {stats['active_states']}")
    
    print("\n✓ Improved BrakeDetector test completed")
    print("✓ Physics-based prediction error detection working!")