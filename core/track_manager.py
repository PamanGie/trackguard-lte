"""
Pure Smart Hungarian TrackManager - IoU-Based Implementation
===========================================================

TrackManager yang menggunakan MURNI Smart Hungarian Algorithm dengan IoU Calculator.
Mengganti ReID, Graph, dan GNN dengan IoU-based calculation untuk speed optimization.

FEATURES:
- Pure Smart Hungarian Algorithm untuk data association
- IoU-based uncertainty calculation (ByteTrack style)
- Quality-controlled assignment dengan 5-layer gate system
- Scene-adaptive thresholds
- Lightweight dan efficient (target 30+ FPS)

PIPELINE:
YOLOv8 → IoU Calculator → Smart Hungarian → TrackUpdate
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import time
import cv2
from collections import defaultdict
import logging

# Import pipeline components
from core.detector import YOLOv8Detector
from core.iou_calculator import IoUCalculator
from core.smart_hungarian import SmartHungarianOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalmanFilter:
    """Simple Kalman filter untuk motion prediction"""
    
    def __init__(self):
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.covariance = np.eye(4) * 1000
        
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.Q = np.array([
            [0.1, 0, 0, 0],
            [0, 0.1, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 8]
        ])
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.R = np.array([
            [20, 0],
            [0, 20]
        ])
    
    def predict(self):
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state[:2]
    
    def update(self, measurement):
        y = measurement - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance


class Track:
    """Track representation untuk Pure Smart Hungarian TrackManager"""
    
    def __init__(self, track_id: int, detection: Dict, features: np.ndarray, frame_id: int):
        self.track_id = track_id
        self.state = 'active'
        
        # Current state
        self.current_detection = detection
        self.current_features = features
        self.current_frame = frame_id
        
        # Track history
        self.history = [detection]
        self.feature_history = [features]
        self.frame_history = [frame_id]
        
        # Track statistics
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.last_seen = frame_id
        
        # Ghost node specific
        self.ghost_start_frame = None
        self.ghost_predictions = []
        self.kalman_filter = KalmanFilter()
        self.ghost_confidence = 1.0
        self.initial_confidence = detection.get('confidence', 0.8)
        
        # Initialize Kalman filter
        center = detection['center']
        self.kalman_filter.state[:2] = center
        
        # Motion model
        self.velocity = np.array([0.0, 0.0])
        
        # Track quality metrics
        self.avg_confidence = self.initial_confidence
        self.confidence_history = [self.initial_confidence]
        self.stability_score = 1.0
        
        # Smart Hungarian specific attributes
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.confidence = self.initial_confidence
        self.time_since_update = 0
    
    def update(self, detection: Dict, features: np.ndarray, frame_id: int):
        """Update track with new detection"""
        # Update motion model
        if len(self.history) >= 2:
            prev_center = self.current_detection['center']
            new_center = detection['center']
            new_velocity = np.array([
                new_center[0] - prev_center[0],
                new_center[1] - prev_center[1]
            ])
            
            alpha = 0.7
            self.velocity = alpha * self.velocity + (1 - alpha) * new_velocity
        
        # Update Kalman filter
        self.kalman_filter.update(np.array(detection['center']))
        
        # Update state
        self.current_detection = detection
        self.current_features = features
        self.current_frame = frame_id
        self.last_seen = frame_id
        
        # Update history
        self.history.append(detection)
        self.feature_history.append(features)
        self.frame_history.append(frame_id)
        
        # Update confidence tracking
        current_conf = detection.get('confidence', 0.8)
        self.confidence_history.append(current_conf)
        self.avg_confidence = np.mean(self.confidence_history[-5:])
        
        # Update stability score
        self._update_stability_score()
        
        # Update statistics
        self.age += 1
        self.hits += 1
        self.misses = 0
        self.time_since_update = 0
        self.state = 'active'
        
        # Update Smart Hungarian attributes
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.confidence = current_conf
        
        # Reset ghost state
        if self.state in ['ghost', 'lost']:
            self.ghost_start_frame = None
            self.ghost_predictions = []
            self.ghost_confidence = 1.0
    
    def _update_stability_score(self):
        """Calculate track stability based on motion consistency"""
        if len(self.history) < 3:
            self.stability_score = 1.0
            return
        
        recent_positions = [det['center'] for det in self.history[-5:]]
        if len(recent_positions) < 3:
            return
        
        velocities = []
        for i in range(1, len(recent_positions)):
            vel = [
                recent_positions[i][0] - recent_positions[i-1][0],
                recent_positions[i][1] - recent_positions[i-1][1]
            ]
            velocities.append(vel)
        
        if len(velocities) >= 2:
            velocity_std = np.std(velocities, axis=0)
            velocity_consistency = 1.0 / (1.0 + np.mean(velocity_std))
            self.stability_score = 0.7 * velocity_consistency + 0.3 * self.avg_confidence
    
    def miss(self, frame_id: int, enable_ghost: bool = True):
        """Mark track as missed in current frame"""
        self.age += 1
        self.misses += 1
        self.time_since_update += 1
        self.current_frame = frame_id
        
        if enable_ghost and self.state == 'active':
            if self.hits >= 3 and self.stability_score > 0.5:
                self.state = 'ghost'
                self.ghost_start_frame = frame_id
                self.ghost_confidence = min(0.9, self.stability_score)
            else:
                self.state = 'lost'
        elif self.state == 'ghost':
            self.ghost_confidence *= 0.8
            
        # Termination conditions
        max_missing = 20 if self.hits >= 5 else 15
        if self.misses >= max_missing:
            self.state = 'terminated'
        elif not enable_ghost and self.state != 'ghost':
            self.state = 'lost'
    
    def get_average_features(self, window_size: int = 5) -> np.ndarray:
        """Get average features over recent frames"""
        if not self.feature_history:
            return np.zeros(128, dtype=np.float32)
        
        recent_features = self.feature_history[-window_size:]
        if len(recent_features) == 1:
            return recent_features[0]
        
        weights = np.exp(np.linspace(-1, 0, len(recent_features)))
        weights = weights / np.sum(weights)
        
        weighted_features = np.zeros_like(recent_features[0])
        for i, features in enumerate(recent_features):
            weighted_features += weights[i] * features
        
        return weighted_features


class PureSmartHungarianTrackManager:
    """
    Pure Smart Hungarian TrackManager
    
    Simplified TrackManager yang menggunakan MURNI Smart Hungarian Algorithm:
    - Detection Pipeline (YOLOv8)
    - Feature Extraction Pipeline (MobileNetV3)
    - Graph Construction Pipeline (GraphBuilder)
    - GNN Prediction Pipeline (GATTracker)
    - Pure Smart Hungarian Data Association Pipeline
    - Track Management Pipeline
    """
    
    def __init__(self, config=None, use_physics=False):
        """Initialize Pure Smart Hungarian TrackManager"""
        from utils.settings import SETTINGS
        
        # Use centralized settings
        if config is None:
            tracking_config = SETTINGS.get_tracking_config()
        else:
            if hasattr(config, 'tracking'):
                tracking_config = config.tracking.__dict__
            else:
                tracking_config = config
        
        # Basic tracking parameters
        # Note: reid_threshold removed - using IoU-based approach instead
        # Note: max_missing_frames removed - using max_age from config instead
        self.min_track_length = tracking_config.get('min_hits', 3)  # Use min_hits from TRACKING_CONFIG
        
        # === LTE-TrackGuard Physics Integration ===
        self.use_physics = use_physics
        self.physics_config = SETTINGS.PHYSICS_CONFIG if use_physics else None
        self.behaviour_detectors = None
        self.physics_predictor = None
        self.eager_smoother = None
        self.velocity_field = None
        
        if use_physics:
            print("🔬 LTE-TrackGuard Physics Mode ENABLED")
            print(f"   EAGER smoothing: {self.physics_config['enable_eager']}")
            print(f"   Physics predictor: {self.physics_config['enable_physics_predictor']}")
            print(f"   Behaviour detection: {self.physics_config['enable_behaviour_detection']}")
            
            # Initialize foundation modules (Phase 1)
            from physics.velocity_field import VelocityField
            self.velocity_field = VelocityField(self.physics_config['velocity_field'])
            print("   ✓ VelocityField loaded")
            
            # Initialize scene analyzer (Phase 4)
            from physics.scene_analyzer import SceneAnalyzer
            self.scene_analyzer = SceneAnalyzer(self.physics_config['scene_analyzer'])
            print("   ✓ SceneAnalyzer loaded")
            
            if self.physics_config['enable_eager']:
                from physics.eager import EAGERSmoother
                self.eager_smoother = EAGERSmoother(self.physics_config['eager'])
                print("   ✓ EAGER Smoother loaded")
            
            # Initialize behaviour detectors (Phase 2-3)
            if self.physics_config['enable_behaviour_detection']:
                from physics.fallen_detector import FallenDetector
                from physics.turn_detector import TurnDetector
                from physics.brake_detector import BrakeDetector
                from physics.collision_detector import CollisionDetector
                
                self.behaviour_detectors = {
                    'fallen': FallenDetector(self.physics_config['fallen_detector']),
                    'turn': TurnDetector(self.physics_config['turn_detector']),
                    'brake': BrakeDetector(self.physics_config['brake_detector']),
                    'collision': CollisionDetector(self.physics_config['collision_detector'])
                }
                print("   ✓ All Behaviour Detectors loaded (fallen, turn, brake, collision)")
            
            # Placeholder untuk behaviour detector results (Phase 2+)
            self.behaviour_results = {
                'turn': [],
                'brake': [],
                'fallen': [],
                'collision': []
            }
        else:
            print("📊 Standard Kalman Mode (default)")
        
        # Track storage
        self.tracks = {}
        self.next_track_id = 1
        self.confirmed_tracks = {}
        
        # === INITIALIZE PIPELINE COMPONENTS ===
        print("Initializing Pure Smart Hungarian TrackManager pipeline components...")
        
        # Detection Pipeline Component
        self.detector = YOLOv8Detector()
        print("✓ YOLOv8 Detection Pipeline initialized")
        
        # IoU Calculation Pipeline Component
        self.iou_calculator = IoUCalculator()
        print("✓ IoU Calculator Pipeline initialized")
        
        # Pure Smart Hungarian Data Association Component
        smart_hungarian_config = {
            'max_distance': 150.0,
            'max_motion_error': 80.0,
            'min_detection_confidence': 0.3,
            'min_track_stability': 0.2,
            'motion_consistency_age': 3
        }
        self.smart_hungarian_optimizer = SmartHungarianOptimizer(smart_hungarian_config)
        print("✓ Pure Smart Hungarian Data Association Pipeline initialized")
        
        # Statistics
        self.total_tracks_created = 0
        self.total_associations = 0
        self.total_id_switches = 0
        self.total_ghost_reidentifications = 0
        
        # Performance tracking
        self.frame_count = 0
        self.pipeline_stats = {
            'detection_time': [],
            'iou_calculation_time': [],
            'data_association_time': [],
            'track_update_time': [],
            'total_pipeline_time': []
        }
        
        # Pure Smart Hungarian statistics
        self.smart_hungarian_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'avg_assignment_rate': 0.0,
            'avg_execution_time': 0.0,
            'quality_controlled_assignments': 0,
            'rejected_assignments': 0
        }
        
        print("🎯 Pure Smart Hungarian TrackManager initialized")
        print("   Pipeline: YOLOv8 → IoU Calculator → Smart Hungarian → TrackUpdate")

    def process_frame(self, image: np.ndarray, frame_id: int) -> Dict:
        """
        MAIN PIPELINE METHOD - Process satu frame lengkap dengan IoU-based Smart Hungarian
        """
        self.frame_count += 1
        pipeline_start_time = time.time()

        # R2Q4 per-frame split timing populated by CollisionDetector.
        self._frame_ppl_compute_time = 0.0
        self._frame_alarm_decision_time = 0.0
        self._frame_collision_detector_time = 0.0
        
        logger.info(f"Processing frame {frame_id} - IoU-based Smart Hungarian Pipeline")
        
        # === DETECTION PIPELINE ===
        detection_start = time.time()
        detections = self._detection_pipeline(image)
        detection_time = time.time() - detection_start
        self.pipeline_stats['detection_time'].append(detection_time)
        
        # === IoU CALCULATION PIPELINE ===
        iou_start = time.time()
        # IoU Calculator dipanggil dengan tracks yang akan digunakan Smart Hungarian
        active_tracks = [track for track in self.tracks.values() if track.state == 'active']
        iou_features = self.iou_calculator.calculate_combined_features(active_tracks, detections)
        iou_time = time.time() - iou_start
        self.pipeline_stats['iou_calculation_time'].append(iou_time)
        
        # === PURE SMART HUNGARIAN DATA ASSOCIATION PIPELINE ===
        association_start = time.time()
        association_results = self._pure_smart_hungarian_association_pipeline(
            detections, iou_features, frame_id
        )
        association_time = time.time() - association_start
        self.pipeline_stats['data_association_time'].append(association_time)
        
        # === TRACK UPDATE PIPELINE ===
        update_start = time.time()
        update_results = self._track_update_pipeline(association_results, detections, iou_features, frame_id)
        update_time = time.time() - update_start
        self.pipeline_stats['track_update_time'].append(update_time)
        
        # Total pipeline time
        total_pipeline_time = time.time() - pipeline_start_time
        self.pipeline_stats['total_pipeline_time'].append(total_pipeline_time)
        
        # Compile complete results
        pipeline_results = {
            'frame_id': frame_id,
            'detections': detections,
            'iou_features': iou_features,
            'active_tracks': self.get_current_tracks(),
            'association_results': association_results,
            'update_results': update_results,
            'pipeline_timing': {
                'detection_time': detection_time,
                'iou_calculation_time': iou_time,
                'data_association_time': association_time,
                'track_update_time': update_time,
                'tracking_time': max(
                    0.0, iou_time + association_time + update_time
                    - self._frame_collision_detector_time
                ),
                'physics_time': self._frame_collision_detector_time,
                'ppl_time': self._frame_ppl_compute_time,
                'alarm_decision_time': self._frame_alarm_decision_time,
                'total_pipeline_time': total_pipeline_time
            },
            'pipeline_stats': self._get_pipeline_stats()
        }
        
        logger.info(f"Frame {frame_id} processed: {len(detections)} detections → {len(self.get_current_tracks())} tracks")
        
        return pipeline_results
    
    def _detection_pipeline(self, image: np.ndarray) -> List[Dict]:
        """Detection Pipeline - YOLOv8 pedestrian detection"""
        logger.debug("Running Detection Pipeline (YOLOv8)")
        
        detections = self.detector.detect(image)
        
        logger.debug(f"Detection Pipeline: {len(detections)} detections found")
        return detections
    
    
    def _pure_smart_hungarian_association_pipeline(self, detections: List[Dict], iou_features: Dict, frame_id: int) -> Dict:
        """Pure Smart Hungarian Data Association Pipeline dengan IoU features"""
        logger.debug("Running Pure Smart Hungarian Data Association Pipeline")
        
        if len(detections) == 0:
            return self._handle_empty_detections(frame_id)
        
        active_tracks = [t for t in self.tracks.values() if t.state in ['active', 'ghost']]
        
        if len(active_tracks) == 0:
            return self._handle_no_tracks(detections, iou_features, frame_id)
        
        # Separate active dan ghost tracks
        active_only_tracks = [t for t in active_tracks if t.state == 'active']
        ghost_tracks = [t for t in active_tracks if t.state == 'ghost']
        
        # Pure Smart Hungarian Association untuk active tracks
        smart_hungarian_associations = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = []
        
        if active_only_tracks:
            smart_hungarian_result = self._pure_smart_hungarian_association(
                active_only_tracks, detections, iou_features
            )
            
            smart_hungarian_associations = smart_hungarian_result['associations']
            unmatched_tracks = smart_hungarian_result['unmatched_tracks']
            
            # Update unmatched detections
            matched_det_indices = [assoc['detection_idx'] for assoc in smart_hungarian_associations]
            unmatched_detections = [i for i in range(len(detections)) 
                                  if i not in matched_det_indices]
        
        # Ghost track association menggunakan IoU features
        ghost_associations = self._associate_ghost_tracks(
            ghost_tracks, detections, iou_features, unmatched_detections
        )
        
        # Merge hasil
        smart_hungarian_associations.extend(ghost_associations['associations'])
        for det_idx in ghost_associations['matched_detections']:
            if det_idx in unmatched_detections:
                unmatched_detections.remove(det_idx)
        
        unmatched_tracks.extend(ghost_associations['unmatched_ghost_tracks'])
        
        logger.debug(f"Pure Smart Hungarian Pipeline: {len(smart_hungarian_associations)} associations made")
        
        return {
            'associations': smart_hungarian_associations,
            'unmatched_tracks': unmatched_tracks,
            'unmatched_detections': unmatched_detections,
            'active_tracks': active_tracks,
            'ghost_associations': len(ghost_associations['associations']),
            'smart_hungarian_info': smart_hungarian_result.get('pure_info', {}) if active_only_tracks else {}
        }
    
    def _pure_smart_hungarian_association(self, tracks: List[Track], detections: List[Dict], 
                                        iou_features: Dict) -> Dict:
        """
        Pure Smart Hungarian Association dengan IoU features
        
        Simplified uncertainty calculation menggunakan IoU Calculator
        """
        start_time = time.time()
        self.smart_hungarian_stats['total_calls'] += 1
        
        try:
            # STEP 1: Use IoU-based uncertainty matrix
            uncertainty_matrix = iou_features['cost_matrix']
            
            logger.debug(f"IoU-based uncertainty matrix: {uncertainty_matrix.shape}")
            logger.debug(f"   Range: {np.min(uncertainty_matrix):.3f} - {np.max(uncertainty_matrix):.3f}")
            logger.debug(f"   Mean: {np.mean(uncertainty_matrix):.3f}")
            logger.debug(f"   Tracks: {len(tracks)}, Detections: {len(detections)}")
            
            # CRITICAL: Validate matrix dimensions before passing to Smart Hungarian
            if uncertainty_matrix.shape != (len(tracks), len(detections)):
                logger.warning(f"Matrix dimension mismatch - rebuilding IoU features!")
                logger.warning(f"   Matrix shape: {uncertainty_matrix.shape}")
                logger.warning(f"   Expected: ({len(tracks)}, {len(detections)})")
                logger.warning(f"   Recalculating IoU features with correct dimensions...")
                
                # Recalculate IoU features with current tracks and detections
                iou_features = self.iou_calculator.calculate_combined_features(tracks, detections)
                uncertainty_matrix = iou_features['cost_matrix']
                
                # Validate again
                if uncertainty_matrix.shape != (len(tracks), len(detections)):
                    logger.error(f"CRITICAL: Still dimension mismatch after recalculation!")
                    logger.error(f"   Matrix shape: {uncertainty_matrix.shape}")
                    logger.error(f"   Expected: ({len(tracks)}, {len(detections)})")
                    return {'associations': [], 'unmatched_tracks': tracks, 'unmatched_detections': list(range(len(detections)))}
            
            # STEP 2: Smart Hungarian Quality-Controlled Assignment
            hungarian_assignments, hungarian_info = self.smart_hungarian_optimizer.optimize_assignment(
                uncertainty_matrix, tracks, detections
            )
            
            # STEP 2.5: Validate assignments before processing
            valid_assignments = []
            for track_idx, det_idx in hungarian_assignments:
                if (0 <= track_idx < len(tracks) and 
                    0 <= det_idx < len(detections) and
                    track_idx < uncertainty_matrix.shape[0] and 
                    det_idx < uncertainty_matrix.shape[1]):
                    valid_assignments.append((track_idx, det_idx))
                else:
                    logger.warning(f"Filtering invalid assignment ({track_idx}, {det_idx}) - bounds: tracks={len(tracks)}, detections={len(detections)}, matrix={uncertainty_matrix.shape}")
            
            hungarian_assignments = valid_assignments
            
            self.smart_hungarian_stats['quality_controlled_assignments'] += len(hungarian_assignments)
            
            logger.debug(f"Pure Smart Hungarian assignments: {len(hungarian_assignments)}")
            logger.debug(f"   Algorithm: {hungarian_info.get('algorithm', 'pure_smart_hungarian')}")
            logger.debug(f"   Assignment rate: {hungarian_info.get('assignment_rate', 0):.1%}")
            
            # STEP 3: Convert results ke TrackGraph format
            associations = []
            matched_track_indices = []
            
            for track_idx, det_idx in hungarian_assignments:
                try:
                    # Double-check bounds (should be safe after validation)
                    if (0 <= track_idx < len(tracks) and 0 <= det_idx < len(detections) and 
                        track_idx < uncertainty_matrix.shape[0] and det_idx < uncertainty_matrix.shape[1]):
                        
                        track = tracks[track_idx]
                        uncertainty_cost = uncertainty_matrix[track_idx, det_idx]
                        
                        associations.append({
                            'track_id': track.track_id,
                            'detection_idx': det_idx,
                            'cost': uncertainty_cost,
                            'type': 'pure_smart_hungarian_iou',
                            'uncertainty': uncertainty_cost,
                            'quality_controlled': True
                        })
                        
                        matched_track_indices.append(track_idx)
                        self.total_associations += 1
                    else:
                        logger.warning(f"Skipping invalid assignment ({track_idx}, {det_idx}) - bounds: tracks={len(tracks)}, detections={len(detections)}, matrix={uncertainty_matrix.shape}")
                except (IndexError, KeyError) as e:
                    logger.warning(f"Error processing assignment ({track_idx}, {det_idx}): {e}")
                    continue
            
            # Unmatched tracks
            unmatched_tracks = [tracks[i] for i in range(len(tracks)) 
                              if i not in matched_track_indices]
            
            # Update statistics
            execution_time = time.time() - start_time
            assignment_rate = len(associations) / min(len(tracks), len(detections)) if tracks and detections else 0
            
            self.smart_hungarian_stats['successful_calls'] += 1
            self.smart_hungarian_stats['avg_assignment_rate'] = (
                (self.smart_hungarian_stats['avg_assignment_rate'] * (self.smart_hungarian_stats['successful_calls'] - 1) + 
                assignment_rate) / self.smart_hungarian_stats['successful_calls']
            )
            self.smart_hungarian_stats['avg_execution_time'] = (
                (self.smart_hungarian_stats['avg_execution_time'] * (self.smart_hungarian_stats['successful_calls'] - 1) + 
                execution_time) / self.smart_hungarian_stats['successful_calls']
            )
            
            logger.debug(f"Pure Smart Hungarian success: {len(associations)} assignments, rate: {assignment_rate:.1%}")
            
            return {
                'associations': associations,
                'unmatched_tracks': unmatched_tracks,
                'pure_info': {
                    'assignment_rate': assignment_rate,
                    'execution_time': execution_time,
                    'algorithm_used': 'pure_smart_hungarian_iou_based',
                    'optimization_info': hungarian_info,
                    'uncertainty_matrix_shape': uncertainty_matrix.shape,
                    'pure_smart_hungarian_active': True,
                    'quality_controlled': True,
                    'assignments_proposed': hungarian_info.get('assignments_proposed', len(hungarian_assignments)),
                    'assignments_accepted': len(hungarian_assignments),
                    'scene_analysis': hungarian_info.get('scene_analysis', {}),
                    'quality_info': hungarian_info.get('quality_info', {})
                }
            }
            
        except Exception as e:
            logger.error(f"Pure Smart Hungarian failed: {e}")
            
            return {
                'associations': [],
                'unmatched_tracks': tracks,
                'pure_info': {
                    'assignment_rate': 0.0,
                    'execution_time': time.time() - start_time,
                    'error': str(e),
                    'algorithm_used': 'pure_smart_hungarian_iou_failed'
                }
            }
    
    def _associate_ghost_tracks(self, ghost_tracks: List[Track], detections: List[Dict], 
                              iou_features: Dict, available_detections: List[int]) -> Dict:
        """Ghost track association menggunakan IoU features"""
        associations = []
        matched_detections = []
        unmatched_ghost_tracks = []
        
        # Get IoU matrix from features
        iou_matrix = iou_features.get('iou_matrix', np.array([]))
        
        if iou_matrix.size == 0:
            return {
                'associations': [],
                'matched_detections': [],
                'unmatched_ghost_tracks': ghost_tracks
            }
        
        for ghost_track in ghost_tracks:
            best_match = None
            best_iou = 0
            best_det_idx = -1
            
            # Find best IoU match for this ghost track
            ghost_track_idx = None
            for i, track in enumerate(self.tracks.values()):
                if track.track_id == ghost_track.track_id:
                    ghost_track_idx = i
                    break
            
            if ghost_track_idx is not None and ghost_track_idx < iou_matrix.shape[0]:
                for det_idx in available_detections:
                    if det_idx < iou_matrix.shape[1] and det_idx < len(detections):
                        iou = iou_matrix[ghost_track_idx, det_idx]
                        
                        # Spatial constraint for ghost tracks
                        detection = detections[det_idx]
                        spatial_distance = np.sqrt(
                            (ghost_track.center[0] - detection['center'][0])**2 +
                            (ghost_track.center[1] - detection['center'][1])**2
                        )
                        
                        max_ghost_distance = 150.0
                        min_ghost_iou = 0.3
                        
                        if (iou > best_iou and 
                            iou > min_ghost_iou and
                            spatial_distance < max_ghost_distance):
                            
                            best_iou = iou
                            best_match = detection
                            best_det_idx = det_idx
            
            if best_match is not None:
                associations.append({
                    'track_id': ghost_track.track_id,
                    'detection_idx': best_det_idx,
                    'cost': 1.0 - best_iou,
                    'type': 'ghost_reidentification_iou',
                    'iou': best_iou
                })
                matched_detections.append(best_det_idx)
                self.total_ghost_reidentifications += 1
            else:
                unmatched_ghost_tracks.append(ghost_track)
        
        return {
            'associations': associations,
            'matched_detections': matched_detections,
            'unmatched_ghost_tracks': unmatched_ghost_tracks
        }
    
    def _track_update_pipeline(self, association_results: Dict, detections: List[Dict], 
                             iou_features: Dict, frame_id: int) -> Dict:
        """Track Update Pipeline - Update track states dan lifecycle management"""
        logger.debug("Running Track Update Pipeline")
        
        ghost_reidentifications = 0
        smart_hungarian_associations = 0
        
        # Update associated tracks
        for assoc in association_results['associations']:
            track_id = assoc['track_id']
            det_idx = assoc['detection_idx']
            
            track = self.tracks[track_id]
            detection = detections[det_idx]
            
            if assoc.get('type') == 'ghost_reidentification_iou':
                ghost_reidentifications += 1
            elif assoc.get('type') == 'pure_smart_hungarian_iou':
                smart_hungarian_associations += 1
            
            # Update track dengan detection (tanpa features karena menggunakan IoU)
            track.update(detection, None, frame_id)
        
        # === LTE-TrackGuard: EAGER Smoothing (Phase 1) ===
        if self.use_physics and self.eager_smoother is not None:
            # Smooth all active tracks
            active_tracks = [t for t in self.tracks.values() if t.state == 'active']
            self.eager_smoother.smooth_batch(active_tracks)
        
        # === LTE-TrackGuard: Scene Analysis (Phase 4) ===
        scene_analysis = None
        if self.use_physics and hasattr(self, 'scene_analyzer'):
            active_tracks = [t for t in self.tracks.values() if t.state == 'active']
            
            # Analyze scene - need image shape, get from first detection if available
            if len(detections) > 0 and 'bbox' in detections[0]:
                # Estimate frame shape from bbox (rough approximation)
                frame_height = 1080  # Default, will be updated
                frame_width = 1920
                
                scene_analysis = self.scene_analyzer.analyze_scene(
                    active_tracks, 
                    (frame_height, frame_width)
                )
                
                logger.debug(f"Scene: {scene_analysis['category']} "
                           f"({scene_analysis['density']:.1f} tracks/Mpx)")
        
        # === LTE-TrackGuard: Behaviour Detection (Phase 2-3) ===
        if self.use_physics and self.behaviour_detectors is not None:
            active_tracks = [t for t in self.tracks.values() if t.state == 'active']
            
            # Apply adaptive thresholds jika scene analysis available
            original_thresholds = None
            if scene_analysis is not None:
                original_thresholds = self.scene_analyzer.apply_adaptive_thresholds(
                    self.behaviour_detectors,
                    scene_analysis['adaptive_params']
                )
            
            # Run all behaviour detectors
            # IMPORTANT: Run collision detector FIRST, then brake/turn, then fallen detector
            # This allows collision to suppress brake/turn (collision > brake severity)
            for detector_name, detector in self.behaviour_detectors.items():
                if detector_name == 'collision':
                    # Pass scene density ke PPL scalar log (per-frame update)
                    if scene_analysis is not None and hasattr(detector, 'update_scene_info'):
                        detector.update_scene_info(
                            scene_analysis.get('density', 0.0),
                            scene_analysis.get('category', 'unknown')
                        )
                    # Run collision detector FIRST
                    detector.total_tracks_created_so_far = self.total_tracks_created
                    _collision_profile_start = time.perf_counter()
                    results = detector.detect(
                        active_tracks,
                        self.velocity_field
                    )
                    _collision_elapsed = time.perf_counter() - _collision_profile_start
                    _ppl_elapsed = float(getattr(detector, 'last_ppl_compute_time', 0.0))
                    _decision_elapsed = float(getattr(
                        detector, 'last_alarm_decision_time',
                        max(0.0, _collision_elapsed - _ppl_elapsed)
                    ))
                    _detector_elapsed = float(getattr(
                        detector, 'last_collision_detector_time', _collision_elapsed
                    ))
                    self._frame_ppl_compute_time += _ppl_elapsed
                    self._frame_alarm_decision_time += _decision_elapsed
                    self._frame_collision_detector_time += _detector_elapsed
                    self.behaviour_results[detector_name] = results
                    
                    # Get collision track IDs untuk suppress brake/turn
                    collision_track_ids = set()
                    for det in results:
                        collision_track_ids.add(det.get('track_id', -1))
                        collision_track_ids.add(det.get('track_id_secondary', -1))
                    
                elif detector_name == 'fallen':
                    # Run fallen detector LAST, with brake/turn results for validation
                    brake_results = self.behaviour_results.get('brake', [])
                    turn_results = self.behaviour_results.get('turn', [])
                    
                    # DEBUG: Log brake/turn results sebelum fallen detection
                    # Log setiap frame untuk debugging (hanya log jika ada detections atau setiap 30 frames)
                    if len(brake_results) > 0 or len(turn_results) > 0 or frame_id % 30 == 0:
                        logger.warning(f"[DEBUG] Frame {frame_id}: Brake detections: {len(brake_results)}, "
                                     f"Turn detections: {len(turn_results)}, Active tracks: {len(active_tracks)}")
                        if len(brake_results) > 0:
                            brake_track_ids = [b.get('track_id') for b in brake_results]
                            logger.warning(f"[DEBUG] Brake track IDs: {brake_track_ids}")
                        if len(turn_results) > 0:
                            turn_track_ids = [t.get('track_id') for t in turn_results]
                            logger.warning(f"[DEBUG] Turn track IDs: {turn_track_ids}")
                    
                    results = detector.detect(
                        active_tracks, 
                        self.velocity_field,
                        brake_results=brake_results,
                        turn_results=turn_results
                    )
                    self.behaviour_results[detector_name] = results
                else:
                    # Run brake, turn detectors normally
                    results = detector.detect(active_tracks, self.velocity_field)
                    
                    # Suppress brake/turn jika track involved in collision (even in monitoring state)
                    if detector_name in ['brake', 'turn']:
                        # Check collision pairs (including monitoring state)
                        collision_pairs = getattr(self.behaviour_detectors.get('collision'), 'collision_pairs', {})
                        collision_track_ids = set()
                        
                        # Get all track IDs involved in collision (monitoring or confirmed)
                        for pair_key, pair_data in collision_pairs.items():
                            if pair_data.get('state') in ['monitoring', 'confirmed']:
                                track_id_1, track_id_2 = pair_key
                                collision_track_ids.add(track_id_1)
                                collision_track_ids.add(track_id_2)
                        
                        # Also check confirmed collision results
                        collision_results = self.behaviour_results.get('collision', [])
                        for det in collision_results:
                            collision_track_ids.add(det.get('track_id', -1))
                            collision_track_ids.add(det.get('track_id_secondary', -1))
                        
                        # Filter out brake/turn detections untuk tracks involved in collision
                        if len(collision_track_ids) > 0:
                            filtered_results = []
                            for det in results:
                                track_id = det.get('track_id', -1)
                                if track_id not in collision_track_ids:
                                    filtered_results.append(det)
                                else:
                                    logger.warning(f"[SUPPRESS] Frame {frame_id}: Suppressed {detector_name} for Track {track_id} (involved in collision)")
                            
                            results = filtered_results
                    
                    self.behaviour_results[detector_name] = results
                
                # Log critical detections
                if len(results) > 0:
                    for det in results:
                        severity = det.get('severity', 'unknown')
                        behaviour = det.get('behaviour_type', detector_name)
                        
                        if severity in ['high', 'critical']:
                            logger.warning(f"🚨 {behaviour.upper()}: Track {det['track_id']} "
                                         f"at frame {frame_id} (severity: {severity})")
                        elif severity == 'medium':
                            logger.info(f"⚠️  {behaviour}: Track {det['track_id']} at frame {frame_id}")
            
            # Restore original thresholds
            if original_thresholds is not None:
                self.scene_analyzer.restore_thresholds(
                    self.behaviour_detectors,
                    original_thresholds
                )
        
        # Handle unmatched tracks
        for track in association_results['unmatched_tracks']:
            from utils.settings import SETTINGS
            enable_ghost = SETTINGS.get_tracking_config().get('enable_ghost_nodes', True)
            track.miss(frame_id, enable_ghost=enable_ghost)
        
        # Create new tracks
        new_tracks = []
        for det_idx in association_results['unmatched_detections']:
            detection = detections[det_idx]
            
            min_confidence = 0.4
            if detection.get('confidence', 0.8) >= min_confidence:
                new_track = self._create_new_track(detection, None, frame_id)
                new_tracks.append(new_track)
        
        # Cleanup terminated tracks
        terminated_tracks = self._cleanup_tracks()
        
        # Update confirmed tracks
        self._update_confirmed_tracks()
        
        logger.debug(f"Track Update Pipeline: {len(association_results['associations'])} updated, "
                    f"{len(new_tracks)} new, {len(terminated_tracks)} terminated")
        
        return {
            'updated_tracks': len(association_results['associations']),
            'new_tracks': len(new_tracks),
            'terminated_tracks': len(terminated_tracks),
            'ghost_reidentifications': ghost_reidentifications,
            'smart_hungarian_associations': smart_hungarian_associations,
            'ghost_associations': association_results.get('ghost_associations', 0),
            'total_active_tracks': len([t for t in self.tracks.values() if t.state == 'active']),
            'total_ghost_tracks': len([t for t in self.tracks.values() if t.state == 'ghost']),
            'smart_hungarian_info': association_results.get('smart_hungarian_info', {})
        }
    
    
    def _create_new_track(self, detection: Dict, features: Optional[np.ndarray], frame_id: int) -> Track:
        """Create new track dengan IoU-based approach"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        # Create track tanpa features (menggunakan IoU)
        new_track = Track(track_id, detection, None, frame_id)
        self.tracks[track_id] = new_track
        
        self.total_tracks_created += 1
        
        logger.debug(f"Created new track {track_id}")
        return new_track
    
    def _cleanup_tracks(self) -> List[Track]:
        """Remove terminated tracks"""
        terminated_tracks = []
        
        track_ids_to_remove = []
        for track_id, track in self.tracks.items():
            if track.state == 'terminated':
                terminated_tracks.append(track)
                track_ids_to_remove.append(track_id)
        
        for track_id in track_ids_to_remove:
            del self.tracks[track_id]
            if track_id in self.confirmed_tracks:
                del self.confirmed_tracks[track_id]
        
        return terminated_tracks
    
    def _update_confirmed_tracks(self):
        """Update confirmed tracks based on track quality"""
        for track_id, track in self.tracks.items():
            if (track.hits >= self.min_track_length and 
                getattr(track, 'stability_score', 0.5) > 0.6 and
                track_id not in self.confirmed_tracks):
                self.confirmed_tracks[track_id] = track
    
    def _handle_empty_detections(self, frame_id: int) -> Dict:
        """Handle case with no detections"""
        unmatched_tracks = []
        
        for track in self.tracks.values():
            if track.state in ['active', 'ghost']:
                from utils.settings import SETTINGS
                enable_ghost = SETTINGS.get_tracking_config().get('enable_ghost_nodes', True)
                track.miss(frame_id, enable_ghost=enable_ghost)
                unmatched_tracks.append(track)
        
        return {
            'associations': [],
            'unmatched_tracks': unmatched_tracks,
            'unmatched_detections': [],
            'active_tracks': [],
            'ghost_associations': 0,
            'smart_hungarian_info': {'assignment_rate': 0.0, 'execution_time': 0.0, 'algorithm_used': 'no_detections'}
        }
    
    def _handle_no_tracks(self, detections: List[Dict], iou_features: Dict, frame_id: int) -> Dict:
        """Handle case with no existing tracks"""
        return {
            'associations': [],
            'unmatched_tracks': [],
            'unmatched_detections': list(range(len(detections))),
            'active_tracks': [],
            'ghost_associations': 0,
            'smart_hungarian_info': {'assignment_rate': 0.0, 'execution_time': 0.0, 'algorithm_used': 'no_tracks'}
        }
    
    def _get_pipeline_stats(self) -> Dict:
        """Get comprehensive pipeline statistics"""
        if not self.pipeline_stats['total_pipeline_time']:
            return {'frames_processed': 0}
        
        recent_frames = 20  # Stats untuk 20 frames terakhir
        stats = {}
        
        for stage, times in self.pipeline_stats.items():
            recent_times = times[-recent_frames:] if len(times) > recent_frames else times
            if recent_times:
                stats[f'{stage}_avg'] = np.mean(recent_times)
                stats[f'{stage}_std'] = np.std(recent_times)
        
        # Calculate FPS
        recent_total_times = self.pipeline_stats['total_pipeline_time'][-recent_frames:]
        if recent_total_times:
            avg_total_time = np.mean(recent_total_times)
            stats['fps'] = 1.0 / avg_total_time if avg_total_time > 0 else 0
        
        stats['frames_processed'] = self.frame_count
        
        return stats
    
    # === PUBLIC INTERFACE METHODS ===
    
    def get_current_tracks(self, confirmed_only: bool = False) -> List[Track]:
        """Get current active tracks"""
        if confirmed_only:
            return [track for track in self.confirmed_tracks.values() 
                   if track.state == 'active']
        else:
            return [track for track in self.tracks.values() 
                   if track.state == 'active']
    
    def get_track_statistics(self) -> Dict:
        """Get comprehensive tracking statistics"""
        active_tracks = len([t for t in self.tracks.values() if t.state == 'active'])
        confirmed_tracks = len(self.confirmed_tracks)
        ghost_tracks = len([t for t in self.tracks.values() if t.state == 'ghost'])
        
        base_stats = {
            'total_tracks_created': self.total_tracks_created,
            'active_tracks': active_tracks,
            'ghost_tracks': ghost_tracks,
            'confirmed_tracks': confirmed_tracks,
            'total_associations': self.total_associations,
            'total_id_switches': self.total_id_switches,
            'total_ghost_reidentifications': self.total_ghost_reidentifications,
            'frames_processed': self.frame_count
        }
        
        # Add Pure Smart Hungarian statistics
        base_stats.update({
            'smart_hungarian_stats': self.smart_hungarian_stats,
            'smart_hungarian_success_rate': (self.smart_hungarian_stats['successful_calls'] / 
                                           max(1, self.smart_hungarian_stats['total_calls'])),
            'matching_algorithm': 'pure_smart_hungarian',
            'pipeline_orchestrator': 'pure_smart_hungarian_trackmanager'
        })
        
        # LTE-TrackGuard: Add physics behaviour stats if enabled
        if self.use_physics and self.physics_config.get('enable_behaviour_detection', False):
            base_stats['physics_behaviour_detection'] = {
                'turn_detections': len(self.behaviour_results['turn']),
                'brake_detections': len(self.behaviour_results['brake']),
                'fallen_detections': len(self.behaviour_results['fallen']),
                'collision_detections': len(self.behaviour_results['collision'])
            }
        
        # Add pipeline performance stats
        base_stats['pipeline_performance'] = self._get_pipeline_stats()
                
        return base_stats
    
    def get_smart_hungarian_performance_summary(self) -> Dict:
        """Get detailed Pure Smart Hungarian performance summary"""
        if self.smart_hungarian_stats['total_calls'] == 0:
            return {'available': False, 'reason': 'no_calls'}
        
        # Get optimizer performance stats
        optimizer_stats = self.smart_hungarian_optimizer.get_performance_statistics()
        
        return {
            'available': True,
            'total_calls': self.smart_hungarian_stats['total_calls'],
            'successful_calls': self.smart_hungarian_stats['successful_calls'],
            'success_rate': self.smart_hungarian_stats['successful_calls'] / self.smart_hungarian_stats['total_calls'],
            'avg_assignment_rate': self.smart_hungarian_stats['avg_assignment_rate'],
            'avg_execution_time': self.smart_hungarian_stats['avg_execution_time'],
            'quality_controlled_assignments': self.smart_hungarian_stats['quality_controlled_assignments'],
            'rejected_assignments': self.smart_hungarian_stats['rejected_assignments'],
            'architecture': 'pure_smart_hungarian_quality_controlled',
            'orchestrator': 'pure_smart_hungarian_trackmanager',
            'optimizer_performance': optimizer_stats if optimizer_stats.get('available', False) else {}
        }
    
    def visualize_tracks(self, image: np.ndarray, 
                        show_confirmed_only: bool = False,
                        show_trajectories: bool = True,
                        show_ghost_tracks: bool = True) -> np.ndarray:
        """Visualize tracks dengan Pure Smart Hungarian performance info"""
        vis_image = image.copy()
        
        tracks_to_show = self.get_current_tracks(confirmed_only=show_confirmed_only)
        ghost_tracks = [t for t in self.tracks.values() if t.state == 'ghost']
        
        # Draw active tracks
        for track in tracks_to_show:
            if track.state != 'ghost':
                color = self._get_track_color(track.track_id)
                
                bbox = track.current_detection['bbox']
                x1, y1, x2, y2 = bbox
                
                thickness = 3 if track.track_id in self.confirmed_tracks else 2
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
                
                # Label dengan Pure Smart Hungarian info
                confidence = track.current_detection.get('confidence', 0.8)
                label = f"PSH_ID:{track.track_id} ({confidence:.2f}) H:{track.hits}"
                if hasattr(track, 'stability_score'):
                    label += f" S:{track.stability_score:.2f}"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 5), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(vis_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw trajectory
                if show_trajectories and len(track.history) > 1:
                    centers = [det['center'] for det in track.history[-10:]]
                    for i in range(1, len(centers)):
                        pt1 = (int(centers[i-1][0]), int(centers[i-1][1]))
                        pt2 = (int(centers[i][0]), int(centers[i][1]))
                        cv2.line(vis_image, pt1, pt2, color, 2)
                        cv2.circle(vis_image, pt2, 3, color, -1)
        
        # Hapus semua info text yang menampilkan karakter aneh
        # Info sudah ada di panel detection summary di main.py
        
        return vis_image
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for track ID"""
        np.random.seed(track_id)
        color = tuple(np.random.randint(50, 255, 3).tolist())
        return color


# BACKWARD COMPATIBILITY & ALIASES
PureTrackManager = PureSmartHungarianTrackManager
PureSmartHungarianManager = PureSmartHungarianTrackManager


# TESTING
if __name__ == "__main__":
    print("🎯 Pure Smart Hungarian TrackManager")
    print("=" * 60)
    
    # Test initialization
    track_manager = PureSmartHungarianTrackManager()
    
    # Print pipeline info
    print(f"\n🔧 Pure Pipeline Architecture:")
    print(f"  Detection: YOLOv8Detector")
    print(f"  Feature Extraction: MobileNetV3Extractor")
    print(f"  Graph Construction: GraphBuilder")
    print(f"  GNN Prediction: GATTracker")
    print(f"  Data Association: Pure Smart Hungarian (Quality Controlled)")
    print(f"  Track Management: Enhanced lifecycle")
    
    # Create dummy frame untuk test
    dummy_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    print(f"\n🧪 Testing pure pipeline with dummy frame...")
    
    try:
        # Test pipeline
        results = track_manager.process_frame(dummy_frame, frame_id=1)
        
        print(f"\n🎯 Pure Smart Hungarian Pipeline Results:")
        print(f"  Frame: {results['frame_id']}")
        print(f"  Detections: {len(results['detections'])}")
        print(f"  Active tracks: {len(results['active_tracks'])}")
        print(f"  Pipeline timing: {results['pipeline_timing']}")
        
        # Get statistics
        stats = track_manager.get_track_statistics()
        pure_hungarian_perf = track_manager.get_smart_hungarian_performance_summary()
        
        print(f"\n🎯 Pure Smart Hungarian Statistics:")
        print(f"  Total tracks created: {stats['total_tracks_created']}")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Pure Smart Hungarian success rate: {stats['smart_hungarian_success_rate']:.2%}")
        print(f"  Pipeline orchestrator: {stats['pipeline_orchestrator']}")
        
        if pure_hungarian_perf['available']:
            print(f"\n🚀 Pure Smart Hungarian Performance:")
            print(f"  Architecture: {pure_hungarian_perf['architecture']}")
            print(f"  Quality controlled assignments: {pure_hungarian_perf['quality_controlled_assignments']}")
        
        print("\n✅ Pure Smart Hungarian TrackManager test completed")
        print("🎯 Ready for pure implementation!")
        print("🔧 Evaluator usage: track_manager.process_frame(image, frame_id)")
        
    except Exception as e:
        print(f"\n❌ Pure Smart Hungarian pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 PURE IMPLEMENTATION SUMMARY:")
    print("1. NO AGM dependency - Pure Smart Hungarian only")
    print("2. Simplified uncertainty calculation with 5 factors:")
    print("   - IoU uncertainty")
    print("   - Distance uncertainty")  
    print("   - Feature similarity uncertainty")
    print("   - Motion prediction uncertainty")
    print("   - Confidence uncertainty")
    print("3. Smart Hungarian quality-controlled assignment")
    print("4. 5-layer quality gate system")
    print("5. Scene-adaptive thresholds")
    print("6. Expected improvements:")
    print("   - ID Switch: -30% hingga -50%")
    print("   - MOTA: maintained atau +1-3%")
    print("   - Simplified implementation")
    print("   - No complex AGM fallbacks")
    print("=" * 60)
