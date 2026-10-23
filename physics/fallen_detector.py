"""
Fallen Detector for LTE-TrackGuard
===================================

Deteksi motor jatuh menggunakan:
1. Rotation angle change (>70 degrees)
2. Aspect ratio flip (vertical → horizontal)
3. Sudden velocity drop (>90%)

From Blueprint Section 2.4.4: Behaviour 3 - Deteksi Motor Jatuh
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from physics.base_detector import BaseDetector, BehaviourState
from physics.fallen_memory import FallenMemory
from physics.fallen_primitive_layer import FallenPrimitiveLayer

logger = logging.getLogger(__name__)


class FallenDetector(BaseDetector):
    """
    Detector untuk motor jatuh
    
    Physics metrics:
    - Rotation: Δθ > 70° dalam <1 detik
    - Aspect Ratio flip: AR > 1.2 → AR < 0.8
    - Sudden stop: velocity drop >90%
    """
    
    def __init__(self, config: Dict):
        """
        Initialize fallen detector
        
        Args:
            config: Configuration dari PHYSICS_CONFIG['fallen_detector']
        """
        super().__init__(config, "FallenDetector")
        
        # Thresholds dari blueprint
        self.rotation_threshold = config.get('rotation_threshold', 70.0)  # degrees
        self.ar_standing_min = config.get('ar_standing_min', 1.2)  # h/w > 1.2
        self.ar_fallen_max = config.get('ar_fallen_max', 0.8)    # h/w < 0.8
        self.velocity_drop_ratio = config.get('velocity_drop_ratio', 0.1)  # <10% remaining
        
        # Use higher persistence untuk avoid false alarms
        self.persist_threshold = config.get('persist_threshold', 0.8)  # 80%

        # Ablation flags: nonaktifkan metrik tertentu untuk single-metric evaluation
        self.enable_ar_flip     = config.get('enable_ar_flip', True)
        self.enable_rotation    = config.get('enable_rotation', True)
        self.enable_velocity_drop = config.get('enable_velocity_drop', True)
        
        # Physics prediction parameters
        self.enable_physics_prediction = config.get('enable_physics_prediction', True)
        self.friction_coefficient = config.get('friction_coefficient', 0.95)
        self.max_prediction_frames = config.get('max_prediction_frames', 60)
        self.confidence_decay_rate = config.get('confidence_decay_rate', 0.97)
        self.min_prediction_confidence = config.get('min_prediction_confidence', 0.3)
        self.fallen_memory = {}  # {track_id: FallenMemory} - Memory untuk fallen objects

        # ============================================
        # FALLEN PHYSICS PRIMITIVE LAYER (MONET)
        # Single-track PPL for motorcycle fall evidence. This runs before the
        # legacy score heuristic and keeps scalar logs for ablation.
        # ============================================
        ppl_config = config.get('fallen_ppl', config.get('ppl', {}))
        self.fallen_ppl = FallenPrimitiveLayer(fps=30.0, config=ppl_config)
        self.enable_fallen_ppl = ppl_config.get('enabled', True)
        self.enable_legacy_fallback = ppl_config.get('enable_legacy_fallback', True)
        
        print(f"  Rotation threshold: {self.rotation_threshold}°")
        print(f"  AR standing min: {self.ar_standing_min}")
        print(f"  AR fallen max: {self.ar_fallen_max}")
        print(f"  Velocity drop ratio: {self.velocity_drop_ratio}")
        if self.enable_physics_prediction:
            print(f"  Physics prediction: ENABLED")
            print(f"    Friction coefficient: {config.get('friction_coefficient', 0.95)}")
            print(f"    Max prediction frames: {config.get('max_prediction_frames', 60)}")

    def update_fps(self, fps: float):
        """Update FPS-dependent windows for fallen PPL."""
        self.fallen_ppl.update_fps(fps)

    def reset_ppl(self):
        """Reset fallen PPL scalar state between videos."""
        self.fallen_ppl.reset()

    def enable_scalar_logging(self, video_name: str = "", label: int = 0):
        """Enable scalar logging for fallen PPL ablation."""
        self.fallen_ppl.enable_scalar_logging(video_name, label)

    def disable_scalar_logging(self):
        """Disable scalar logging for fallen PPL."""
        self.fallen_ppl.disable_scalar_logging()

    def get_scalar_log(self):
        """Return fallen PPL scalar records collected so far."""
        return self.fallen_ppl.get_scalar_log()
    
    def _compute_metric(self, track: object, velocity_field: 'VelocityField') -> float:
        """
        Compute fallen metric (combined score)
        
        Returns combined score [0, 3] dari 3 metrics:
        - 1.0 jika rotation anomaly
        - 1.0 jika AR flip
        - 1.0 jika sudden stop
        - 1.0 jika AR sudah rendah (motor sudah jatuh dari awal)
        
        Args:
            track: Track object
            velocity_field: VelocityField untuk velocity calculation
            
        Returns:
            Combined metric score
        """
        score = 0.0
        
        # Only detect motorcycles
        class_name = track.current_detection.get('class_name', 'unknown')
        if class_name != 'motorcycle':
            return 0.0
        
        # SIMPLE CHECK: Jika AR sudah rendah, langsung deteksi sebagai fallen
        # Ini untuk kasus motor sudah jatuh dari awal (video sintetis)
        current_bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
        current_ar = self._compute_aspect_ratio(current_bbox)
        
        # STRICT: Hanya deteksi sebagai fallen jika benar-benar jatuh
        # Bukan hanya karena angle kamera miring (motor menepi)
        
        # Metric 1: Rotation angle change
        rotation_score = self._check_rotation_anomaly(track) if self.enable_rotation else 0.0
        if rotation_score > 0:
            score += 1.0

        # Metric 2: Aspect ratio flip
        ar_flip_score = self._check_ar_flip(track) if self.enable_ar_flip else 0.0
        score += ar_flip_score

        # Metric 3: Sudden velocity drop
        velocity_drop_score = self._check_sudden_stop(track, velocity_field) if self.enable_velocity_drop else 0.0
        score += velocity_drop_score

        # Low-AR bonus (hanya aktif jika ar_flip diaktifkan)
        if self.enable_ar_flip and current_ar < self.ar_fallen_max:
            # Jika AR sangat rendah (< 0.6), kemungkinan besar jatuh
            if current_ar < 0.6:
                score += 1.5  # AR sangat rendah, kemungkinan besar jatuh
            # Jika AR 0.6-0.7, kemungkinan jatuh
            elif current_ar < 0.7:
                score += 1.0  # AR rendah, kemungkinan jatuh
            # Jika AR 0.7-0.9, kemungkinan jatuh (tapi lebih rendah)
            else:
                score += 0.5  # AR sedang, kemungkinan jatuh (lebih rendah)
        
        # Motor yang benar-benar jatuh bisa punya:
        # 1. Rotation signifikan (>70°) - score +1.0
        # 2. AR flip (standing → fallen) - score +1.0
        # 3. Velocity drop >90% - score +1.0
        # 4. AR sangat rendah (< 0.6) - score +1.5
        # Validasi kecepatan normal akan filter motor yang masih jalan normal
        return score
    
    def _check_condition(self, metric_value: float, track: object) -> bool:
        """
        Check apakah fallen condition terpenuhi
        
        Threshold metric >= 1.0 untuk deteksi fallen
        Validasi kecepatan normal akan filter motor yang masih jalan normal
        
        Args:
            metric_value: Combined metric score
            track: Track object
            
        Returns:
            True jika fallen detected
        """
        # Threshold 1.0 untuk memastikan motor yang benar-benar jatuh tetap terdeteksi
        # Validasi kecepatan normal akan filter motor yang masih jalan normal
        return metric_value >= 1.0
    
    def detect(self, tracks: List, velocity_field: 'VelocityField', 
               brake_results: List[Dict] = None, turn_results: List[Dict] = None) -> List[Dict]:
        """
        Main detection method dengan physics prediction untuk occlusion handling
        
        Flow:
        1. Detect fallen dari active tracks (YOLO deteksi) - dengan brake/turn validation
        2. Predict fallen objects yang YOLO tidak deteksi (physics prediction)
        
        Args:
            tracks: List of track objects
            velocity_field: VelocityField untuk physics calculations
            brake_results: Results from BrakeDetector untuk prevent false positive
            turn_results: Results from TurnDetector untuk prevent false positive
            
        Returns:
            List of confirmed detections (confirmed + predicted)
        """
        detections = []
        
        # Default empty lists jika tidak ada
        if brake_results is None:
            brake_results = []
        if turn_results is None:
            turn_results = []
        
        # Get current frame ID (dari track jika ada)
        current_frame = tracks[0].current_frame if tracks and hasattr(tracks[0], 'current_frame') else 0
        
        # ============================================
        # STEP 1: DETECT fallen dari active tracks (YOLO deteksi)
        # ============================================
        active_track_ids = set()
        
        for track in tracks:
            # Skip tracks yang tidak valid
            if not self._is_valid_track(track):
                continue
            
            # HANYA process motorcycle, skip person atau class lain
            class_name = track.current_detection.get('class_name', 'unknown')
            if class_name != 'motorcycle':
                continue  # Skip person dan class lain
            
            active_track_ids.add(track.track_id)

            # ============================================
            # FALLEN PHYSICS PRIMITIVE LAYER (PPL)
            # Priority path: if single-track physics evidence fires, report a
            # fall event and bypass the legacy heuristic for this track.
            # ============================================
            ppl_result = None
            if self.enable_fallen_ppl:
                ppl_result = self.fallen_ppl.check_track(
                    track,
                    current_frame,
                    nearby_tracks=tracks
                )
            if ppl_result is not None:
                _scalar = ppl_result.get('scalar', {})
                logger.debug(
                    f"[PPL FALLEN] Frame {current_frame}: Track {track.track_id} | "
                    f"Path={ppl_result['path_triggered']} | "
                    f"Primitives={'+'.join(ppl_result['primitives_fired'])} | "
                    f"confidence={ppl_result['confidence']:.2f} | "
                    f"KCR={_scalar.get('kcr_value', 0.0):.3f} "
                    f"KCRz={_scalar.get('kcr_z', 0.0):.2f} "
                    f"ARz={_scalar.get('ar_z', 0.0):.2f} "
                    f"ARdrop={_scalar.get('ar_drop', 0.0):.2f} "
                    f"PPS={_scalar.get('pps_score', 0.0):.2f} "
                    f"SDE={_scalar.get('sde_score', 0.0):.2f} "
                    f"cornerMob={_scalar.get('corner_mobility', 0.0):.2f} "
                    f"cornerCos={_scalar.get('corner_heading_cos', 0.0):.2f}"
                )
                detection_result = self._create_ppl_detection_result(track, ppl_result)
                detections.append(detection_result)

                if self.enable_physics_prediction:
                    self._create_or_update_fallen_memory(track, velocity_field, current_frame)
                continue

            if self.enable_fallen_ppl and not self.enable_legacy_fallback:
                continue
            
            # ============================================
            # VALIDATION: Check apakah track sedang brake, turn, atau decelerating
            # Jika ya, REJECT fallen detection (prevent false positive)
            # ============================================
            track_braking = any(b.get('track_id') == track.track_id for b in brake_results)
            track_turning = any(t.get('track_id') == track.track_id for t in turn_results)
            
            # Check deceleration: velocity drop tapi masih bergerak = motor melambat, bukan jatuh
            is_decelerating = self._check_deceleration(track, velocity_field)
            
            if track_braking or track_turning or is_decelerating:
                # REJECT: Motor sedang rem, belok, atau melambat, bukan jatuh
                # TAPI: Buat detection result untuk decelerating untuk visualisasi
                # (class_name sudah di-check di awal loop, jadi pasti motorcycle)
                if is_decelerating and not track_braking and not track_turning:
                    # Create decelerating detection untuk visualisasi (hanya motorcycle)
                    decel_detection = {
                        'track_id': track.track_id,
                        'frame_id': current_frame,
                        'detector': self.detector_name,
                        'behaviour_type': 'decelerating',
                        'bbox': track.current_detection.get('bbox', [0, 0, 0, 0]),
                        'center': track.current_detection.get('center', [0, 0]),
                        'prediction_mode': 'confirmed',
                        'severity': 'medium',
                        'alert_level': 'info',
                        'metric_value': 0.0,
                        'persistence': 0.5
                    }
                    detections.append(decel_detection)
                
                # Log rejection
                if track_braking:
                    reason = "braking"
                elif track_turning:
                    reason = "turning"
                else:
                    reason = "decelerating"
                logger.debug(f"[FALLEN REJECT] Track {track.track_id} "
                             f"(reason: {reason}) - preventing false positive")
                continue
            
            # Compute physics metric
            metric_value = self._compute_metric(track, velocity_field)
            
            # ============================================
            # VALIDATION: Check apakah motor masih bergerak normal
            # Motor yang masih jalan normal di aspal tidak boleh terdeteksi sebagai fallen
            # meskipun AR rendah (angle kamera miring)
            # ============================================
            # Check kecepatan normal - jika motor masih bergerak dengan kecepatan normal,
            # kemungkinan besar tidak jatuh (meskipun AR rendah)
            v_current = velocity_field.compute_velocity(track, dt=1.0)
            speed_current = np.linalg.norm(v_current)
            normal_speed_threshold = 5.0  # px/frame - kecepatan normal untuk motor jalan
            
            # Check metrics untuk pastikan motor benar-benar jatuh
            rotation_score = self._check_rotation_anomaly(track)
            ar_flip_score = self._check_ar_flip(track)
            velocity_drop_score = self._check_sudden_stop(track, velocity_field)
            
            # REJECT jika motor masih bergerak normal TAPI:
            # 1. Tidak ada rotation signifikan, DAN
            # 2. Tidak ada AR flip, DAN
            # 3. Tidak ada velocity drop, DAN
            # 4. AR tidak terlalu rendah (>= 0.5) - jika AR < 0.5, kemungkinan besar jatuh meskipun speed tinggi
            current_bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
            current_ar = self._compute_aspect_ratio(current_bbox)
            
            is_moving_normally = speed_current > normal_speed_threshold
            has_no_fallen_indicators = (rotation_score == 0 and 
                                       ar_flip_score == 0 and 
                                       velocity_drop_score == 0)
            ar_not_too_low = current_ar >= 0.5  # AR >= 0.5 = tidak terlalu rendah
            
            # REJECT hanya jika semua kondisi terpenuhi:
            # - Masih bergerak normal (> 5 px/frame)
            # - Tidak ada indikator jatuh (rotation/AR flip/velocity drop)
            # - AR tidak terlalu rendah (>= 0.5)
            if is_moving_normally and has_no_fallen_indicators and ar_not_too_low:
                # Motor masih bergerak normal tanpa indikator jatuh dan AR tidak terlalu rendah = tidak jatuh
                # Skip fallen detection untuk track ini
                logger.debug(f"[FALLEN REJECT] Track {track.track_id} "
                             f"(reason: moving normally without fallen indicators, "
                             f"speed={speed_current:.2f} px/frame, AR={current_ar:.2f})")
                continue
            
            # Check condition
            condition_met = self._check_condition(metric_value, track)
            
            # SIMPLIFIED: Untuk FallenDetector dengan metric tinggi, langsung confirm
            # Bypass state machine untuk kasus motor sudah jelas jatuh
            if metric_value >= 1.5:
                # Langsung create detection tanpa state machine
                detection_result = self._create_detection_result(track, metric_value, 1.0)
                detection_result['prediction_mode'] = 'confirmed'  # YOLO confirmed
                detections.append(detection_result)
                
                # Create/update FallenMemory untuk physics prediction
                if self.enable_physics_prediction:
                    self._create_or_update_fallen_memory(track, velocity_field, current_frame)
                continue
            
            # Update state machine
            detection_result = self._update_state_machine(
                track.track_id, 
                condition_met, 
                metric_value,
                track
            )
            
            # Collect confirmed detections
            if detection_result is not None:
                detection_result['prediction_mode'] = 'confirmed'  # YOLO confirmed
                detections.append(detection_result)
                
                # Create/update FallenMemory untuk physics prediction
                if self.enable_physics_prediction:
                    self._create_or_update_fallen_memory(track, velocity_field, current_frame)
        
        # ============================================
        # STEP 2: PREDICT fallen objects yang YOLO tidak deteksi (INERTIA!)
        # ============================================
        if self.enable_physics_prediction:
            # Predict dari fallen_memory yang tidak ada di active tracks
            for track_id, memory in list(self.fallen_memory.items()):
                # Skip jika track masih active (sudah dihandle di step 1)
                if track_id in active_track_ids:
                    continue
                
                # ============================================
                # PHYSICS-BASED TERMINATION CHECKS
                # ============================================
                
                # Termination 1: Energy dissipation complete
                # Kalau kinetic energy < 5% dari initial selama 4 frame → kill ghost
                if memory.should_terminate_energy(frames_threshold=4):
                    del self.fallen_memory[track_id]
                    continue
                
                # Termination 2: Static object re-detection
                # Check apakah ada detection baru yang overlap dengan predicted bbox
                # dan memiliki velocity rendah (< 3 km/h ≈ 0.25 px/frame untuk 30fps)
                if self._check_static_re_detection(memory, tracks, velocity_field):
                    del self.fallen_memory[track_id]
                    continue
                
                # Termination 3: Brake/Turn/Deceleration check
                # Jika track yang overlap dengan predicted bbox sedang brake/turn/decelerate,
                # terminate FallenMemory (motor menepi, bukan jatuh)
                if self._check_brake_turn_deceleration_termination(
                    memory, tracks, brake_results, turn_results, velocity_field
                ):
                    logger.warning(f"[DEBUG] FallenDetector: TERMINATED FallenMemory Track {track_id} "
                                 f"(reason: brake/turn/deceleration) - motor menepi, bukan jatuh")
                    del self.fallen_memory[track_id]
                    continue
                
                # Check expiry (time-based)
                if memory.should_expire(current_frame):
                    del self.fallen_memory[track_id]
                    continue
                
                # PHYSICS PREDICTION - Inertia!
                predicted_bbox = memory.predict_next_position()
                
                # Get fall direction untuk visualization
                fall_direction = memory.get_fall_direction()
                vorticity = memory.get_vorticity()
                
                # Create PREDICTED detection
                predicted_detection = {
                    'track_id': track_id,
                    'frame_id': current_frame,
                    'detector': self.detector_name,
                    'behaviour_type': 'motorcycle_fallen',
                    'bbox': predicted_bbox,
                    'center': [
                        (predicted_bbox[0] + predicted_bbox[2]) / 2,
                        (predicted_bbox[1] + predicted_bbox[3]) / 2
                    ],
                    'prediction_mode': 'physics_predicted',  # 🔥 PHYSICS!
                    'confidence': memory.get_confidence(),
                    'frames_since_seen': memory.get_frames_since_seen(current_frame),
                    'velocity': memory.get_velocity().tolist(),
                    'fall_direction': fall_direction.tolist(),  # For direction arrow
                    'vorticity': vorticity,  # For direction sign
                    'severity': 'high',
                    'alert_level': 'emergency',
                    'metric_value': 1.0,  # Default metric untuk predicted
                    'persistence': memory.get_confidence()
                }
                
                detections.append(predicted_detection)
        
        # Cleanup terminated tracks
        self._cleanup_old_states(tracks)
        
        return detections

    def _create_ppl_detection_result(self, track: object, ppl_result: Dict) -> Dict:
        """
        Create fallen detection result from FallenPrimitiveLayer evidence.

        Keeps the legacy output contract (`behaviour_type=motorcycle_fallen`)
        while attaching primitive scalars for MONET ablation.
        """
        scalar = ppl_result.get('scalar', {})
        confidence = float(ppl_result.get('confidence', 1.0))
        result = self._create_detection_result(track, confidence, confidence)
        result.update({
            'prediction_mode': 'confirmed',
            'detection_mode': 'fallen_ppl',
            'detection_tier': 'fallen_ppl',
            'ppl_path': ppl_result.get('path_triggered', 'unknown'),
            'ppl_primitives': '+'.join(ppl_result.get('primitives_fired', [])),
            'ppl_confidence': confidence,
            'ppl_pcs': bool(ppl_result.get('pcs', False)),
            'ppl_kcr': bool(ppl_result.get('kcr', False)),
            'ppl_pps': bool(ppl_result.get('pps', False)),
            'ppl_sde': bool(ppl_result.get('sde', False)),
            'ppl_rcg': bool(ppl_result.get('rcg', False)),
            'ppl_corner_guard': bool(ppl_result.get('corner_guard', False)),
            'ppl_ar_z': float(scalar.get('ar_z', 0.0)),
            'ppl_ar_drop': float(scalar.get('ar_drop', 0.0)),
            'ppl_kcr_value': float(scalar.get('kcr_value', 0.0)),
            'ppl_kcr_z': float(scalar.get('kcr_z', 0.0)),
            'ppl_speed_before': float(scalar.get('speed_before', 0.0)),
            'ppl_speed_after': float(scalar.get('speed_after', 0.0)),
            'ppl_pps_score': float(scalar.get('pps_score', 0.0)),
            'ppl_sde_score': float(scalar.get('sde_score', 0.0)),
            'ppl_sde_total_drop': float(scalar.get('sde_total_drop', 0.0)),
            'ppl_corner_mobility': float(scalar.get('corner_mobility', 0.0)),
            'ppl_corner_heading_cos': float(scalar.get('corner_heading_cos', 0.0)),
        })
        return result
    
    def _create_or_update_fallen_memory(self, track: object, velocity_field: 'VelocityField', 
                                       frame_id: int):
        """
        Create atau update FallenMemory untuk track yang jatuh
        
        Args:
            track: Track object yang jatuh (HARUS motorcycle, sudah di-filter di detect())
            velocity_field: VelocityField untuk compute velocity
            frame_id: Current frame ID
        """
        # DOUBLE CHECK: Pastikan hanya motorcycle yang dibuat FallenMemory
        class_name = track.current_detection.get('class_name', 'unknown')
        if class_name != 'motorcycle':
            logger.warning(f"[DEBUG] FallenDetector: SKIP FallenMemory for Track {track.track_id} "
                         f"(class_name='{class_name}', bukan 'motorcycle')")
            return  # Skip non-motorcycle tracks
        
        # Compute velocity dari velocity_field
        velocity = velocity_field.compute_velocity(track, dt=1.0)
        
        # Compute vorticity untuk direction of fall (dari cross product)
        vorticity = velocity_field.compute_cross_product_2d(track)
        
        # Get bbox
        bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
        
        # Get config untuk FallenMemory (dari self config)
        config = {
            'friction_coefficient': getattr(self, 'friction_coefficient', 0.95),
            'max_prediction_frames': getattr(self, 'max_prediction_frames', 60),
            'confidence_decay_rate': getattr(self, 'confidence_decay_rate', 0.97),
            'min_prediction_confidence': getattr(self, 'min_prediction_confidence', 0.3)
        }
        
        if track.track_id not in self.fallen_memory:
            # New fallen object - create memory
            memory = FallenMemory(track.track_id, bbox, velocity, frame_id, config)
            memory.vorticity = vorticity
            if np.linalg.norm(velocity) > 1e-6:
                memory.fall_direction = velocity / np.linalg.norm(velocity)
            self.fallen_memory[track.track_id] = memory
        else:
            # Update existing memory dengan detection baru
            self.fallen_memory[track.track_id].update_with_detection(
                bbox, velocity, frame_id, vorticity
            )
    
    def _check_brake_turn_deceleration_termination(self, memory: 'FallenMemory', tracks: List,
                                                   brake_results: List[Dict], turn_results: List[Dict],
                                                   velocity_field: 'VelocityField') -> bool:
        """
        Check apakah track yang overlap dengan predicted bbox sedang brake/turn/decelerate
        
        Jika ya, terminate FallenMemory (motor menepi, bukan jatuh)
        
        Args:
            memory: FallenMemory object
            tracks: List of active tracks
            brake_results: Results from BrakeDetector
            turn_results: Results from TurnDetector
            velocity_field: VelocityField untuk compute velocity
            
        Returns:
            True jika harus terminate (brake/turn/deceleration found)
        """
        predicted_bbox = memory.get_predicted_bbox()
        
        # Check semua active tracks yang overlap dengan predicted bbox
        for track in tracks:
            # Only check motorcycle
            class_name = track.current_detection.get('class_name', 'unknown')
            if class_name != 'motorcycle':
                continue
            
            # Compute IoU dengan predicted bbox
            track_bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
            iou = self._compute_iou(predicted_bbox, track_bbox)
            
            if iou > 0.5:  # Tumpang tindih > 50% (lebih longgar untuk catch re-detection)
                # Check brake
                track_braking = any(b.get('track_id') == track.track_id for b in brake_results)
                if track_braking:
                    return True
                
                # Check turn
                track_turning = any(t.get('track_id') == track.track_id for t in turn_results)
                if track_turning:
                    return True
                
                # Check deceleration
                is_decelerating = self._check_deceleration(track, velocity_field)
                if is_decelerating:
                    return True
        
        return False
    
    def _check_static_re_detection(self, memory: 'FallenMemory', tracks: List, 
                                   velocity_field: 'VelocityField') -> bool:
        """
        Check static object re-detection termination
        
        Jika di lokasi ghost tumpang tindih >0.7 IoU dengan detection baru
        class "motorcycle" atau "person" yang kecepatannya <3 km/h → merge & kill ghost
        
        Args:
            memory: FallenMemory object
            tracks: List of active tracks
            velocity_field: VelocityField untuk compute velocity
            
        Returns:
            True jika harus terminate (static re-detection found)
        """
        predicted_bbox = memory.get_predicted_bbox()
        
        # Check semua active tracks
        for track in tracks:
            # Only check motorcycle or person
            class_name = track.current_detection.get('class_name', 'unknown')
            if class_name not in ['motorcycle', 'person']:
                continue
            
            # Compute IoU dengan predicted bbox
            track_bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
            iou = self._compute_iou(predicted_bbox, track_bbox)
            
            if iou > 0.7:  # Tumpang tindih > 70%
                # Check velocity < 3 km/h
                # 3 km/h = 0.833 m/s = 833 mm/s
                # Untuk 30 fps: 833 / 30 = 27.8 mm/frame
                # Asumsi 1 pixel = 10 mm (adjust sesuai video): 27.8 / 10 = 2.78 px/frame
                # Atau lebih simple: velocity < 3 px/frame
                velocity = velocity_field.compute_velocity(track, dt=1.0)
                speed = np.linalg.norm(velocity)  # px/frame
                
                # 3 km/h ≈ 0.25 px/frame untuk typical video (adjust threshold)
                # Atau lebih konservatif: speed < 3 px/frame
                if speed < 3.0:  # Static object (velocity < 3 px/frame)
                    return True  # Terminate ghost
        
        return False
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Compute IoU between two bboxes
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU value [0, 1]
        """
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
        
        if union < 1e-6:
            return 0.0
        
        return intersection / union
    
    def _check_rotation_anomaly(self, track: object) -> float:
        """
        Check rotation angle change (Blueprint Section 2.4.4 - Metric 1)
        
        Δθ = |θ_t - θ_t-5| > 70° dalam <1 detik
        
        Args:
            track: Track object
            
        Returns:
            1.0 jika rotation anomaly, 0.0 otherwise
        """
        # SIMPLIFIED: Kurangi requirement history untuk video pendek
        if not hasattr(track, 'history') or len(track.history) < 3:
            return 0.0
        
        # Compute rotation angle dari velocity direction
        current_angle = self._compute_angle_from_velocity(track)
        
        # Get angle 5 frames ago
        past_angle = self._compute_angle_from_history(track, frames_back=5)
        
        if current_angle is None or past_angle is None:
            return 0.0
        
        # Angle difference (handle wrapping)
        angle_diff = abs(current_angle - past_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # Check threshold
        if angle_diff > self.rotation_threshold:
            return 1.0
        
        return 0.0
    
    def _check_ar_flip(self, track: object) -> float:
        """
        Check aspect ratio flip (Blueprint Section 2.4.4 - Metric 2)
        
        Standing: AR > 1.2 (vertical)
        Fallen: AR < 0.8 (horizontal)
        
        Args:
            track: Track object
            
        Returns:
            1.0 jika AR flip detected, 0.0 otherwise
        """
        # SIMPLIFIED: Kurangi requirement history untuk video pendek
        if not hasattr(track, 'history') or len(track.history) < 3:
            return 0.0
        
        # Current aspect ratio
        current_bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
        current_ar = self._compute_aspect_ratio(current_bbox)
        
        # Past aspect ratio (2 frames ago - lebih dekat)
        past_bbox = track.history[-3].get('bbox', [0, 0, 0, 0])
        past_ar = self._compute_aspect_ratio(past_bbox)
        
        # Check flip: standing → fallen
        if past_ar > self.ar_standing_min and current_ar < self.ar_fallen_max:
            return 1.0
        
        return 0.0
    
    def _check_sudden_stop(self, track: object, velocity_field: 'VelocityField') -> float:
        """
        Check sudden velocity drop (Blueprint Section 2.4.4 - Metric 3)
        
        velocity_ratio = ||v_t|| / ||v_t-3|| < 0.1 (velocity drop >90%)
        
        Args:
            track: Track object
            velocity_field: VelocityField untuk velocity calculation
            
        Returns:
            1.0 jika sudden stop detected, 0.0 otherwise
        """
        # SIMPLIFIED: Kurangi requirement history untuk video pendek
        if not hasattr(track, 'history') or len(track.history) < 3:
            return 0.0
        
        # Current velocity
        v_current = velocity_field.compute_velocity(track, dt=1.0)
        speed_current = np.linalg.norm(v_current)
        
        # SIMPLIFIED: Velocity 2 frames ago (lebih dekat untuk video pendek)
        # Temporarily modify track history untuk compute past velocity
        original_current = track.current_detection
        original_history_len = len(track.history)
        
        # Set position to t-2
        track.current_detection = track.history[-3]
        track.history = track.history[:-2]
        
        v_past = velocity_field.compute_velocity(track, dt=1.0)
        speed_past = np.linalg.norm(v_past)
        
        # Restore track state
        track.current_detection = original_current
        # Restore history (need to add back removed items)
        # This is safe because we're just reading
        
        # Check velocity drop
        if speed_past < 1e-6:  # Avoid division by zero
            return 0.0
        
        velocity_ratio = speed_current / speed_past
        
        if velocity_ratio < self.velocity_drop_ratio:
            return 1.0
        
        return 0.0
    
    def _check_deceleration(self, track: object, velocity_field: 'VelocityField') -> bool:
        """
        Check apakah motor sedang melambat (decelerating) atau menepi tapi masih bergerak
        
        Deceleration = velocity drop (speed_current < speed_past * threshold) 
                      TAPI masih bergerak (speed_current > min_speed)
        
        Motor menepi = AR mungkin rendah (karena angle) TAPI masih bergerak sedikit
        
        Motor yang melambat/menepi bukan jatuh, jadi harus di-reject dari fallen detection.
        
        Args:
            track: Track object
            velocity_field: VelocityField untuk velocity calculation
            
        Returns:
            True jika motor sedang decelerating/menepi, False otherwise
        """
        # Need history untuk compare velocity
        if not hasattr(track, 'history') or len(track.history) < 3:
            return False
        
        # Current velocity
        v_current = velocity_field.compute_velocity(track, dt=1.0)
        speed_current = np.linalg.norm(v_current)
        
        # Past velocity (3 frames ago) - compute dari history langsung
        # Get positions dari history
        current_center = track.current_detection.get('center', [0, 0])
        past_center = track.history[-3].get('center', current_center)
        past_prev_center = track.history[-4].get('center', past_center) if len(track.history) >= 4 else past_center
        
        # Compute past velocity dari history
        v_past = np.array([
            past_center[0] - past_prev_center[0],
            past_center[1] - past_prev_center[1]
        ])
        speed_past = np.linalg.norm(v_past)
        
        # Check deceleration conditions
        if speed_past < 1e-6:  # Avoid division by zero
            return False
        
        # Get current AR untuk check motor menepi
        current_bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
        current_ar = self._compute_aspect_ratio(current_bbox)
        
        # Deceleration: velocity drop > 20% TAPI masih bergerak > 1 px/frame
        velocity_ratio = speed_current / speed_past
        min_speed_threshold = 1.0  # px/frame - masih bergerak sedikit (longgarkan untuk motor menepi)
        deceleration_threshold = 0.8  # velocity drop > 20% (speed_current < 80% speed_past) - lebih sensitif
        
        # Motor melambat/menepi jika:
        # 1. Velocity drop > 20% (speed_current < 80% speed_past) - lebih sensitif
        # 2. TAPI masih bergerak (speed_current > 1 px/frame) - longgarkan threshold
        # 3. Past speed cukup tinggi (speed_past > 2 px/frame) - longgarkan threshold
        is_decelerating = (
            velocity_ratio < deceleration_threshold and  # Velocity drop > 20%
            speed_current > min_speed_threshold and      # Masih bergerak sedikit
            speed_past > 2.0                             # Past speed cukup tinggi
        )
        
        # ATAU: Motor menepi jika AR rendah TAPI masih bergerak sedikit
        # Motor menepi biasanya AR rendah karena angle, tapi masih bergerak
        is_pulling_over = (
            current_ar < self.ar_fallen_max and          # AR rendah (seperti fallen)
            speed_current > 0.5 and                      # Masih bergerak sedikit (> 0.5 px/frame)
            speed_current < 3.0 and                      # Tapi tidak terlalu cepat (< 3 px/frame)
            speed_past > 2.0                             # Past speed cukup tinggi
        )
        
        return is_decelerating or is_pulling_over
    
    def _compute_aspect_ratio(self, bbox: List[float]) -> float:
        """
        Compute aspect ratio dari bbox
        
        AR = height / width
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Aspect ratio
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width < 1e-6:
            return 1.0
        
        return height / width
    
    def _compute_angle_from_velocity(self, track: object) -> Optional[float]:
        """
        Compute orientation angle dari velocity direction
        
        θ = arctan(v_y / v_x)
        
        Args:
            track: Track object
            
        Returns:
            Angle in degrees [0, 360] atau None
        """
        if not hasattr(track, 'history') or len(track.history) < 2:
            return None
        
        # Compute velocity
        current_center = track.current_detection.get('center', [0, 0])
        prev_center = track.history[-2].get('center', current_center)
        
        vx = current_center[0] - prev_center[0]
        vy = current_center[1] - prev_center[1]
        
        # Check if moving
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:
            return None
        
        # Compute angle
        angle_rad = np.arctan2(vy, vx)
        angle_deg = np.degrees(angle_rad)
        
        # Normalize to [0, 360]
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg
    
    def _compute_angle_from_history(self, track: object, frames_back: int = 5) -> Optional[float]:
        """
        Compute angle dari historical velocity
        
        Args:
            track: Track object
            frames_back: How many frames to go back
            
        Returns:
            Angle in degrees [0, 360] atau None
        """
        if len(track.history) < frames_back + 1:
            return None
        
        # Get positions
        center_t = track.history[-(frames_back)].get('center', [0, 0])
        center_t_minus_1 = track.history[-(frames_back + 1)].get('center', center_t)
        
        vx = center_t[0] - center_t_minus_1[0]
        vy = center_t[1] - center_t_minus_1[1]
        
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:
            return None
        
        angle_rad = np.arctan2(vy, vx)
        angle_deg = np.degrees(angle_rad)
        
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg
    
    def _create_detection_result(self, track: object, metric_value: float, 
                                 persistence: float) -> Dict:
        """
        Create fallen detection result dengan detailed info
        
        Args:
            track: Track object
            metric_value: Combined metric score
            persistence: Persistence score
            
        Returns:
            Detection result dict
        """
        result = super()._create_detection_result(track, metric_value, persistence)
        
        # Add fallen-specific info
        bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
        ar_current = self._compute_aspect_ratio(bbox)
        
        result.update({
            'behaviour_type': 'motorcycle_fallen',
            'aspect_ratio_current': float(ar_current),
            'severity': 'high',  # Always high severity untuk fallen
            'alert_level': 'emergency'
        })

        scalar = self.fallen_ppl.get_last_scalar(track.track_id)
        if scalar:
            result.update({
                'detection_mode': 'legacy_fallen',
                'ppl_probe_path': scalar.get('path', 'none'),
                'ppl_primitives': self._format_ppl_probe_primitives(scalar),
                'ppl_pcs': bool(scalar.get('pcs', 0)),
                'ppl_kcr': bool(scalar.get('kcr', 0)),
                'ppl_pps': bool(scalar.get('pps', 0)),
                'ppl_sde': bool(scalar.get('sde', 0)),
                'ppl_rcg': bool(scalar.get('rcg', 0)),
                'ppl_corner_guard': bool(scalar.get('corner_guard', 0)),
                'ppl_ar_z': float(scalar.get('ar_z', 0.0)),
                'ppl_ar_drop': float(scalar.get('ar_drop', 0.0)),
                'ppl_kcr_value': float(scalar.get('kcr_value', 0.0)),
                'ppl_kcr_z': float(scalar.get('kcr_z', 0.0)),
                'ppl_speed_before': float(scalar.get('speed_before', 0.0)),
                'ppl_speed_after': float(scalar.get('speed_after', 0.0)),
                'ppl_pps_score': float(scalar.get('pps_score', 0.0)),
                'ppl_sde_score': float(scalar.get('sde_score', 0.0)),
                'ppl_sde_total_drop': float(scalar.get('sde_total_drop', 0.0)),
                'ppl_corner_mobility': float(scalar.get('corner_mobility', 0.0)),
                'ppl_corner_heading_cos': float(scalar.get('corner_heading_cos', 0.0)),
            })
        
        return result

    def _format_ppl_probe_primitives(self, scalar: Dict) -> str:
        primitives = []
        for key in ('pcs', 'kcr', 'pps', 'sde', 'rcg'):
            if scalar.get(key, 0):
                primitives.append(key.upper())
        if scalar.get('corner_guard', 0):
            primitives.append('CORNER_GUARD')
        return '+'.join(primitives) if primitives else '-'


# Testing
if __name__ == "__main__":
    print("Testing FallenDetector...")
    
    # Create detector
    config = {
        'rotation_threshold': 70.0,
        'ar_standing_min': 1.2,
        'ar_fallen_max': 0.8,
        'velocity_drop_ratio': 0.1,
        'window_size': 5,
        'persist_threshold': 0.8
    }
    
    detector = FallenDetector(config)
    
    # Create dummy track - motorcycle standing then falling
    class DummyTrack:
        def __init__(self):
            self.track_id = 1
            self.state = 'active'
            self.hits = 10
            self.current_frame = 10
            
            # Simulate motorcycle standing (AR > 1.2, moving)
            self.history = [
                {'bbox': [100, 100, 140, 200], 'center': [120, 150]},  # AR = 100/40 = 2.5
                {'bbox': [102, 102, 142, 202], 'center': [122, 152]},
                {'bbox': [104, 104, 144, 204], 'center': [124, 154]},
                {'bbox': [106, 106, 146, 206], 'center': [126, 156]},
                {'bbox': [108, 108, 148, 208], 'center': [128, 158]},
                {'bbox': [110, 110, 150, 210], 'center': [130, 160]},
                # Now falling - AR flip + rotation
                {'bbox': [110, 140, 200, 180], 'center': [155, 160]},  # AR = 40/90 = 0.44
            ]
            
            self.current_detection = self.history[-1]
            self.current_detection['class_name'] = 'motorcycle'
    
    # Create velocity field
    from physics.velocity_field import VelocityField
    vf_config = {'gaussian_radius': 100.0, 'flow_sigma': 50.0}
    vf = VelocityField(vf_config)
    
    # Test detection
    track = DummyTrack()
    
    print("\nTesting fallen detection:")
    print(f"Track class: {track.current_detection['class_name']}")
    print(f"History length: {len(track.history)}")
    
    # Compute metric
    metric = detector._compute_metric(track, vf)
    print(f"Fallen metric: {metric:.2f} (threshold: 2.0)")
    
    # Check condition
    is_fallen = detector._check_condition(metric, track)
    print(f"Fallen detected: {is_fallen}")
    
    # Full detect
    detections = detector.detect([track], vf)
    print(f"\nConfirmed detections: {len(detections)}")
    
    if len(detections) > 0:
        det = detections[0]
        print(f"Detection info:")
        print(f"  Track ID: {det['track_id']}")
        print(f"  Behaviour: {det.get('behaviour_type', 'N/A')}")
        print(f"  Metric value: {det['metric_value']:.2f}")
        print(f"  Persistence: {det['persistence']:.2f}")
        print(f"  Alert level: {det.get('alert_level', 'N/A')}")
    
    # Statistics
    stats = detector.get_statistics()
    print(f"\n✓ FallenDetector Statistics:")
    print(f"  Confirmed detections: {stats['confirmed_detections']}")
    print(f"  Active states: {stats['active_states']}")
    
    print("\n✓ FallenDetector test completed")
