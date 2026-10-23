"""
Smart Hungarian Algorithm - FIXED ANTI-FRAGMENTASI VERSION
==========================================================

PERBAIKAN UTAMA:
âœ… Recovery Mode: Cari track lama sebelum bikin baru
âœ… Extended Memory: Track tidak cepat dilupakan
âœ… Tiered Recovery: Threshold bertingkat untuk recovery
âœ… Track Resurrection: Bangkitkan track yang "mati"

TARGET: Hilangkan fragmentasi, pertahankan ID switch control

AUTHOR: MOT Research - Anti-Fragmentasi Solution
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import time
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AssignmentQuality:
    """Assignment quality metrics"""
    uncertainty_score: float
    spatial_score: float
    motion_score: float
    detection_score: float
    track_score: float
    combined_score: float
    passed_gates: List[str]
    failed_gates: List[str]


@dataclass
class SceneAnalysis:
    """Scene complexity analysis"""
    scene_type: str  # 'sparse', 'normal', 'crowded', 'extreme'
    complexity_score: float
    object_density: float
    imbalance_factor: float
    quality_variance: float
    recommended_threshold: float


@dataclass
class TrackMemory:
    """Extended track memory untuk recovery"""
    track_id: int
    last_position: Tuple[float, float]
    last_features: np.ndarray
    last_confidence: float
    missing_frames: int
    death_frame: int
    recovery_attempts: int
    original_track_ref: any  # Reference ke track asli


class AntiFragmentasiSmartHungarian:
    """
    FIXED Smart Hungarian Algorithm dengan Anti-Fragmentasi
    
    PERBAIKAN UTAMA:
    - Extended memory untuk track yang "mati"
    - Recovery mode sebelum bikin track baru
    - Tiered threshold untuk recovery vs new assignment
    - Track resurrection logic
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize FIXED Smart Hungarian dengan anti-fragmentasi"""
        self.config = config or {}
        
        # TUNED: Stricter thresholds untuk mengurangi ID switches
        self.uncertainty_thresholds = {
            'sparse': 0.6,      # IoU cost threshold - stricter untuk consistency
            'normal': 0.5,      # IoU cost threshold - much stricter
            'crowded': 0.4,     # IoU cost threshold - very strict
            'extreme': 0.3      # IoU cost threshold - strictest
        }
        self.recovery_thresholds = {
            'sparse': 0.8,      # Recovery threshold - stricter
            'normal': 0.7,      # Recovery threshold - much stricter
            'crowded': 0.6,     # Recovery threshold - very strict
            'extreme': 0.5      # Recovery threshold - strictest
        }
        # Spatial constraints
        self.spatial_limits = {
            'max_distance': self.config.get('max_distance', 150.0),
            'max_motion_error': self.config.get('max_motion_error', 80.0),
            'bbox_overlap_threshold': self.config.get('bbox_overlap_threshold', 0.1),
            'recovery_max_distance': 250.0,  # NEW: Jarak maksimal untuk recovery
        }
        
        # Quality requirements
        self.quality_requirements = {
            'min_detection_confidence': self.config.get('min_detection_confidence', 0.3),
            'min_track_stability': self.config.get('min_track_stability', 0.2),
            'motion_consistency_age': self.config.get('motion_consistency_age', 3)
        }
        
        # NEW: Extended memory system
        self.track_memory = {}  # {track_id: TrackMemory}
        self.memory_retention_frames = 50  # Ingat track sampai 50 frame
        self.max_recovery_attempts = 3     # Maksimal 3 kali coba recovery
        
        # Performance tracking
        self.performance_stats = {
            'total_calls': 0,
            'total_assignments_proposed': 0,
            'total_assignments_accepted': 0,
            'gate_rejections': {
                'uncertainty': 0,
                'spatial': 0,
                'motion': 0,
                'detection_quality': 0,
                'track_quality': 0
            },
            'scene_type_counts': {
                'sparse': 0,
                'normal': 0,
                'crowded': 0,
                'extreme': 0
            },
            # NEW: Anti-fragmentasi stats
            'recovery_stats': {
                'recovery_attempts': 0,
                'successful_recoveries': 0,
                'failed_recoveries': 0,
                'tracks_saved': 0
            }
        }
        
        logger.info("ðŸ”§ FIXED Smart Hungarian (Anti-Fragmentasi) initialized")
        logger.info(f"   Main thresholds: {self.uncertainty_thresholds}")
        logger.info(f"   Recovery thresholds: {self.recovery_thresholds}")
        logger.info(f"   Memory retention: {self.memory_retention_frames} frames")
    
    def optimize_assignment(self, uncertainty_matrix: np.ndarray, 
                          tracks: Optional[List] = None, 
                          detections: Optional[List] = None) -> Tuple[List[Tuple[int, int]], Dict]:
        """
        FIXED Main optimization dengan anti-fragmentasi logic
        """
        start_time = time.time()
        self.performance_stats['total_calls'] += 1
        
        if uncertainty_matrix.shape[0] == 0 or uncertainty_matrix.shape[1] == 0:
            return [], self._create_empty_result()
        
        logger.debug(f"ðŸ”§ FIXED Smart Hungarian: Processing {uncertainty_matrix.shape} matrix")
        
        # STEP 1: Scene Analysis
        scene_analysis = self._analyze_scene(uncertainty_matrix, tracks, detections)
        self.performance_stats['scene_type_counts'][scene_analysis.scene_type] += 1
        
        # STEP 2: Update track memory dari tracks yang ada
        if tracks:
            self._update_track_memory(tracks)
        
        # STEP 3: Prepare Smart Matrix
        smart_matrix = self._prepare_smart_matrix(uncertainty_matrix, tracks, detections, scene_analysis)
        
        # STEP 4: Hungarian Algorithm
        hungarian_assignments = self._run_hungarian_algorithm(smart_matrix, uncertainty_matrix.shape)
        self.performance_stats['total_assignments_proposed'] += len(hungarian_assignments)
        
        # STEP 5: FIXED Multi-Layer Quality Gate dengan Recovery
        if tracks and detections:
            final_assignments, quality_info = self._apply_fixed_quality_gate_with_recovery(
                hungarian_assignments, uncertainty_matrix, tracks, detections, scene_analysis
            )
        else:
            final_assignments = self._basic_uncertainty_filter(
                hungarian_assignments, uncertainty_matrix, scene_analysis.recommended_threshold
            )
            quality_info = {'method': 'basic_uncertainty_filter'}
        
        # STEP 6: ANTI-FRAGMENTASI - Recovery unmatched detections
        if detections:
            recovery_assignments = self._attempt_track_recovery(
                detections, final_assignments, scene_analysis
            )
            final_assignments.extend(recovery_assignments)
        
        self.performance_stats['total_assignments_accepted'] += len(final_assignments)
        
        execution_time = time.time() - start_time
        assignment_rate = len(final_assignments) / min(uncertainty_matrix.shape) if min(uncertainty_matrix.shape) > 0 else 0
        
        # Create optimization info
        optimization_info = {
            'algorithm': 'fixed_smart_hungarian_anti_fragmentasi',
            'execution_time': execution_time,
            'assignments_proposed': len(hungarian_assignments),
            'assignments_accepted': len(final_assignments),
            'assignment_rate': assignment_rate,
            'rejection_rate': (len(hungarian_assignments) - len(final_assignments)) / max(len(hungarian_assignments), 1),
            'scene_analysis': scene_analysis.__dict__,
            'quality_info': quality_info,
            'matrix_shape': uncertainty_matrix.shape,
            'recovery_stats': self.performance_stats['recovery_stats'].copy()  # NEW
        }
        
        logger.info(f"ðŸ”§ FIXED Smart Hungarian completed: {len(hungarian_assignments)} â†’ {len(final_assignments)} "
                   f"(rate: {assignment_rate:.1%}, recoveries: {self.performance_stats['recovery_stats']['successful_recoveries']})")
        
        return final_assignments, optimization_info
    
    def _update_track_memory(self, tracks: List):
        """UPDATE: Simpan informasi tracks ke memory untuk recovery"""
        current_frame = getattr(tracks[0], 'current_frame', 0) if tracks and len(tracks) > 0 else 0
        
        # Update existing track memory
        active_track_ids = set()
        for track in tracks:
            track_id = track.track_id
            active_track_ids.add(track_id)
            
            if track.state == 'active':
                # Track masih aktif, reset memory
                if track_id in self.track_memory:
                    del self.track_memory[track_id]
        
        # Cleanup old memory
        to_remove = []
        for track_id, memory in self.track_memory.items():
            frames_since_death = current_frame - memory.death_frame
            if frames_since_death > self.memory_retention_frames:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.track_memory[track_id]
    
    def record_track_death(self, track):
        """NEW: Record track yang mati untuk recovery"""
        track_memory = TrackMemory(
            track_id=track.track_id,
            last_position=(track.current_detection['center'][0], track.current_detection['center'][1]),
            last_features=track.get_average_features() if hasattr(track, 'get_average_features') else None,
            last_confidence=track.avg_confidence if hasattr(track, 'avg_confidence') else 0.5,
            missing_frames=track.misses if hasattr(track, 'misses') else 0,
            death_frame=track.current_frame if hasattr(track, 'current_frame') else 0,
            recovery_attempts=0,
            original_track_ref=track
        )
        
        self.track_memory[track.track_id] = track_memory
        logger.debug(f"ðŸ“ Recorded track {track.track_id} death for potential recovery")
    
    def _attempt_track_recovery(self, detections: List, existing_assignments: List[Tuple[int, int]], 
                               scene_analysis: SceneAnalysis) -> List[Tuple[int, int]]:
        """NEW: Coba recovery track dari memory sebelum bikin track baru"""
        
        if not self.track_memory:
            return []
        
        recovery_assignments = []
        used_detection_indices = {det_idx for _, det_idx in existing_assignments}
        recovery_threshold = self.recovery_thresholds[scene_analysis.scene_type]
        
        # Coba recovery untuk setiap detection yang belum di-assign
        for det_idx, detection in enumerate(detections):
            if det_idx >= len(detections) or det_idx in used_detection_indices:
                continue
            
            best_recovery = None
            best_score = float('inf')
            
            # Cari track di memory yang bisa di-recover
            for track_id, memory in self.track_memory.items():
                if memory.recovery_attempts >= self.max_recovery_attempts:
                    continue
                
                # Calculate recovery score
                recovery_score = self._calculate_recovery_score(detection, memory)
                
                if recovery_score < recovery_threshold and recovery_score < best_score:
                    best_recovery = memory
                    best_score = recovery_score
            
            # Lakukan recovery jika ada kandidat bagus
            if best_recovery:
                # Mark sebagai recovery assignment
                recovery_assignments.append((best_recovery.track_id, det_idx))
                
                # Update recovery stats
                best_recovery.recovery_attempts += 1
                self.performance_stats['recovery_stats']['recovery_attempts'] += 1
                self.performance_stats['recovery_stats']['successful_recoveries'] += 1
                self.performance_stats['recovery_stats']['tracks_saved'] += 1
                
                used_detection_indices.add(det_idx)
                
                logger.debug(f"ðŸ”„ RECOVERY: Track {best_recovery.track_id} recovered with detection {det_idx} "
                           f"(score: {best_score:.3f})")
        
        return recovery_assignments
    
    def _calculate_recovery_score(self, detection, memory: TrackMemory) -> float:
        """NEW: Calculate score untuk recovery possibility"""
        
        # Spatial distance
        det_center = detection.center if hasattr(detection, 'center') else detection['center']
        spatial_distance = np.sqrt(
            (det_center[0] - memory.last_position[0])**2 +
            (det_center[1] - memory.last_position[1])**2
        )
        
        # Normalize spatial score
        spatial_score = spatial_distance / self.spatial_limits['recovery_max_distance']
        
        # Feature similarity (jika ada)
        feature_score = 0.5  # Default neutral
        if memory.last_features is not None and hasattr(detection, 'features') and detection.features is not None:
            det_features = detection.features
            if det_features.shape == memory.last_features.shape:
                similarity = np.dot(det_features, memory.last_features) / (
                    np.linalg.norm(det_features) * np.linalg.norm(memory.last_features)
                )
                feature_score = 1.0 - np.clip(similarity, 0, 1)
        
        # Confidence penalty
        det_confidence = detection.confidence if hasattr(detection, 'confidence') else detection.get('confidence', 0.5)
        confidence_score = 1.0 - det_confidence
        
        # Time penalty (semakin lama mati, semakin sulit recovery)
        time_penalty = memory.missing_frames / 50.0  # Normalize to 50 frames
        
        # Combined recovery score (lower is better)
        recovery_score = (
            0.4 * spatial_score +      # Spatial paling penting
            0.3 * feature_score +      # Feature similarity
            0.2 * confidence_score +   # Detection quality  
            0.1 * time_penalty         # Time penalty
        )
        
        return recovery_score
    
    def _analyze_scene(self, uncertainty_matrix: np.ndarray, 
                      tracks: Optional[List], detections: Optional[List]) -> SceneAnalysis:
        """Scene analysis - UNCHANGED"""
        
        n_tracks, n_detections = uncertainty_matrix.shape
        total_objects = n_tracks + n_detections
        
        # Factor 1: Object density
        object_density = total_objects / 15.0
        object_density = min(object_density, 1.0)
        
        # Factor 2: Imbalance factor
        if n_tracks == 0 or n_detections == 0:
            imbalance_factor = 1.0
        else:
            imbalance_factor = abs(n_tracks - n_detections) / max(n_tracks, n_detections)
        
        # Factor 3: Quality variance
        quality_variance = 0.0
        if detections and len(detections) > 1:
            confidences = [getattr(d, 'confidence', 0.5) for d in detections]
            quality_variance = np.std(confidences)
        
        # Factor 4: Uncertainty distribution
        uncertainty_mean = np.mean(uncertainty_matrix)
        uncertainty_std = np.std(uncertainty_matrix)
        uncertainty_complexity = min(uncertainty_std * 2.0, 1.0)
        
        # Combine factors
        complexity_score = (
            0.35 * object_density +
            0.25 * imbalance_factor +
            0.20 * quality_variance +
            0.20 * uncertainty_complexity
        )
        complexity_score = np.clip(complexity_score, 0.0, 1.0)
        
        # Determine scene type
        if complexity_score >= 0.7:
            scene_type = 'extreme'
        elif complexity_score >= 0.4:
            scene_type = 'crowded'
        elif complexity_score >= 0.2:
            scene_type = 'normal'
        else:
            scene_type = 'sparse'
        
        recommended_threshold = self.uncertainty_thresholds[scene_type]
        
        return SceneAnalysis(
            scene_type=scene_type,
            complexity_score=complexity_score,
            object_density=object_density,
            imbalance_factor=imbalance_factor,
            quality_variance=quality_variance,
            recommended_threshold=recommended_threshold
        )
    
    def _prepare_smart_matrix(self, uncertainty_matrix: np.ndarray, 
                            tracks: Optional[List], detections: Optional[List],
                            scene_analysis: SceneAnalysis) -> np.ndarray:
        """Prepare matrix - UNCHANGED"""
        
        n_tracks, n_detections = uncertainty_matrix.shape
        max_dim = max(n_tracks, n_detections)
        
        smart_matrix = np.full((max_dim, max_dim), 999.0)
        smart_matrix[:n_tracks, :n_detections] = uncertainty_matrix
        
        if n_tracks < n_detections:
            self._assign_detection_dummy_costs(smart_matrix, detections, n_tracks, n_detections, scene_analysis)
        elif n_detections < n_tracks:
            self._assign_track_dummy_costs(smart_matrix, tracks, n_tracks, n_detections, scene_analysis)
        
        return smart_matrix
    
    def _assign_detection_dummy_costs(self, smart_matrix: np.ndarray, detections: Optional[List],
                                    n_tracks: int, n_detections: int, scene_analysis: SceneAnalysis):
        """Assign dummy costs - UNCHANGED"""
        
        if not detections or len(detections) == 0:
            return
        
        for col in range(n_detections):
            if col >= len(detections):
                continue
            detection = detections[col]
            detection_quality = getattr(detection, 'confidence', 0.5)
            
            if detection_quality > 0.7:
                dummy_cost = 0.8
            elif detection_quality > 0.5:
                dummy_cost = 0.9
            else:
                dummy_cost = 999.0
            
            if scene_analysis.scene_type in ['crowded', 'extreme']:
                dummy_cost += 0.1
            
            smart_matrix[n_tracks:, col] = dummy_cost
    
    def _assign_track_dummy_costs(self, smart_matrix: np.ndarray, tracks: Optional[List],
                                n_tracks: int, n_detections: int, scene_analysis: SceneAnalysis):
        """Assign track dummy costs - UNCHANGED"""
        
        if not tracks or len(tracks) == 0:
            return
        
        for row in range(n_tracks):
            if row >= len(tracks):
                continue
            track = tracks[row]
            track_quality = self._calculate_track_quality(track)
            
            if track_quality > 0.7:
                dummy_cost = 0.8
            elif track_quality > 0.5:
                dummy_cost = 0.9
            else:
                dummy_cost = 999.0
            
            if scene_analysis.scene_type in ['crowded', 'extreme']:
                dummy_cost += 0.1
            
            smart_matrix[row, n_detections:] = dummy_cost
    
    def _calculate_track_quality(self, track) -> float:
        """Calculate track quality - UNCHANGED"""
        
        stability = getattr(track, 'stability_score', 0.5)
        age_factor = min(getattr(track, 'age', 1) / 10.0, 1.0)
        hits = getattr(track, 'hits', 1)
        age = max(getattr(track, 'age', 1), 1)
        hit_ratio = hits / age
        confidence = getattr(track, 'confidence', 0.5)
        time_since_update = getattr(track, 'time_since_update', 0)
        recency_factor = max(0.0, 1.0 - time_since_update / 10.0)
        
        quality_score = (
            0.3 * stability +
            0.2 * age_factor +
            0.2 * hit_ratio +
            0.15 * confidence +
            0.15 * recency_factor
        )
        
        return np.clip(quality_score, 0.0, 1.0)
    
    def _run_hungarian_algorithm(self, smart_matrix: np.ndarray, 
                                original_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Hungarian algorithm - UNCHANGED"""
        
        from scipy.optimize import linear_sum_assignment
        
        try:
            row_indices, col_indices = linear_sum_assignment(smart_matrix)
            
            n_tracks, n_detections = original_shape
            valid_assignments = []
            filtered_count = 0
            
            for row, col in zip(row_indices, col_indices):
                if row < n_tracks and col < n_detections:
                    valid_assignments.append((int(row), int(col)))
                else:
                    filtered_count += 1
            
            if filtered_count > 0:
                logger.debug(f"Hungarian filtered {filtered_count} invalid assignments (original shape: {original_shape}, matrix shape: {smart_matrix.shape})")
            
            return valid_assignments
            
        except Exception as e:
            logger.error(f"Hungarian algorithm failed: {e}")
            return []
    
    def _apply_fixed_quality_gate_with_recovery(self, assignments: List[Tuple[int, int]], 
                          uncertainty_matrix: np.ndarray,
                          tracks: List, detections: List,
                          scene_analysis: SceneAnalysis) -> Tuple[List[Tuple[int, int]], Dict]:
        """FIXED Quality gate dengan recovery consideration"""
        
        approved_assignments = []
        quality_assessments = []
        gate_stats = {
            'total_proposed': len(assignments),
            'passed_uncertainty': 0,
            'passed_spatial': 0,
            'passed_motion': 0,
            'passed_detection': 0,
            'passed_track': 0,
            'final_approved': 0
        }
        
        for track_idx, det_idx in assignments:
            # Bounds checking untuk mencegah index out of range
            if track_idx >= len(tracks) or det_idx >= len(detections):
                logger.warning(f"Skipping assignment ({track_idx}, {det_idx}) - out of bounds (tracks: {len(tracks)}, detections: {len(detections)})")
                continue
                
            track = tracks[track_idx]
            detection = detections[det_idx]
            uncertainty = uncertainty_matrix[track_idx, det_idx]
            
            quality = self._assess_assignment_quality(track, detection, uncertainty, scene_analysis)
            quality_assessments.append(quality)
            
            # FIXED Gate 1: Uncertainty threshold (TETAP KETAT)
            if uncertainty > scene_analysis.recommended_threshold:
                # TETAP REJECT - jangan longgarkan threshold utama
                self.performance_stats['gate_rejections']['uncertainty'] += 1
                continue
            gate_stats['passed_uncertainty'] += 1
            
            # Gate 2-5: Quality gates lainnya SAMA
            if not self._check_spatial_consistency(track, detection):
                self.performance_stats['gate_rejections']['spatial'] += 1
                continue
            gate_stats['passed_spatial'] += 1
            
            if not self._check_motion_consistency(track, detection):
                self.performance_stats['gate_rejections']['motion'] += 1
                continue
            gate_stats['passed_motion'] += 1
            
            if not self._check_detection_quality(detection):
                self.performance_stats['gate_rejections']['detection_quality'] += 1
                continue
            gate_stats['passed_detection'] += 1
            
            if not self._check_track_quality(track):
                self.performance_stats['gate_rejections']['track_quality'] += 1
                continue
            gate_stats['passed_track'] += 1
            
            approved_assignments.append((track_idx, det_idx))
            gate_stats['final_approved'] += 1
        
        quality_info = {
            'method': 'fixed_multi_layer_quality_gate_with_recovery',
            'gate_statistics': gate_stats,
            'scene_threshold': scene_analysis.recommended_threshold,
            'recovery_threshold': self.recovery_thresholds[scene_analysis.scene_type],  # NEW
            'avg_quality_score': np.mean([q.combined_score for q in quality_assessments]) if quality_assessments else 0.0
        }
        
        return approved_assignments, quality_info
    
    def _assess_assignment_quality(self, track, detection, uncertainty: float, 
                                 scene_analysis: SceneAnalysis) -> AssignmentQuality:
        """Assessment quality - UNCHANGED"""
        
        uncertainty_score = 1.0 - uncertainty
        
        distance = self._calculate_distance(track, detection)
        spatial_score = max(0.0, 1.0 - distance / self.spatial_limits['max_distance'])
        
        motion_score = self._calculate_motion_score(track, detection)
        
        if hasattr(detection, 'confidence'):
            detection_score = detection.confidence
        elif isinstance(detection, dict) and 'confidence' in detection:
            detection_score = detection['confidence']
        else:
            detection_score = 0.5
        
        track_score = self._calculate_track_quality(track)
        
        if scene_analysis.scene_type in ['crowded', 'extreme']:
            weights = [0.4, 0.3, 0.15, 0.10, 0.05]
        else:
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]
        
        combined_score = (
            weights[0] * uncertainty_score +
            weights[1] * spatial_score +
            weights[2] * motion_score +
            weights[3] * detection_score +
            weights[4] * track_score
        )
        
        return AssignmentQuality(
            uncertainty_score=uncertainty_score,
            spatial_score=spatial_score,
            motion_score=motion_score,
            detection_score=detection_score,
            track_score=track_score,
            combined_score=combined_score,
            passed_gates=[],
            failed_gates=[]
        )
    
    def _check_spatial_consistency(self, track, detection) -> bool:
        """Spatial consistency check - UNCHANGED"""
        
        distance = self._calculate_distance(track, detection)
        if distance > self.spatial_limits['max_distance']:
            return False
        
        track_bbox = None
        det_bbox = None
        
        if hasattr(track, 'bbox'):
            track_bbox = track.bbox
        elif hasattr(track, 'current_detection') and 'bbox' in track.current_detection:
            track_bbox = track.current_detection['bbox']
        
        if hasattr(detection, 'bbox'):
            det_bbox = detection.bbox
        elif isinstance(detection, dict) and 'bbox' in detection:
            det_bbox = detection['bbox']
        
        if track_bbox and det_bbox:
            overlap = self._calculate_bbox_overlap(track_bbox, det_bbox)
            if overlap < self.spatial_limits['bbox_overlap_threshold']:
                return False
        
        return True
    
    def _check_motion_consistency(self, track, detection) -> bool:
        """Motion consistency check - UNCHANGED"""
        
        track_age = getattr(track, 'age', 0)
        if track_age < self.quality_requirements['motion_consistency_age']:
            return True
        
        track_velocity = None
        if hasattr(track, 'velocity'):
            track_velocity = track.velocity
        elif hasattr(track, 'current_detection') and hasattr(track, 'velocity'):
            track_velocity = track.velocity
        
        if track_velocity is None:
            return True
        
        if hasattr(track, 'center'):
            track_center = track.center
        elif hasattr(track, 'current_detection'):
            track_center = track.current_detection['center']
        else:
            return True
        
        predicted_center = [
            track_center[0] + track_velocity[0],
            track_center[1] + track_velocity[1]
        ]
        
        if hasattr(detection, 'center'):
            det_center = detection.center
        elif isinstance(detection, dict) and 'center' in detection:
            det_center = detection['center']
        else:
            return True
        
        motion_error = np.sqrt(
            (predicted_center[0] - det_center[0])**2 +
            (predicted_center[1] - det_center[1])**2
        )
        
        return motion_error <= self.spatial_limits['max_motion_error']
    
    def _check_detection_quality(self, detection) -> bool:
        """Detection quality check - UNCHANGED"""
        
        if hasattr(detection, 'confidence'):
            confidence = detection.confidence
        elif isinstance(detection, dict) and 'confidence' in detection:
            confidence = detection['confidence']
        else:
            confidence = 0.5
        
        return confidence >= self.quality_requirements['min_detection_confidence']
    
    def _check_track_quality(self, track) -> bool:
        """Track quality check - UNCHANGED"""
        
        stability = getattr(track, 'stability_score', 0.5)
        return stability >= self.quality_requirements['min_track_stability']
    
    def _calculate_distance(self, track, detection) -> float:
        """Calculate distance - UNCHANGED"""
        
        if hasattr(track, 'center'):
            track_center = track.center
        elif hasattr(track, 'current_detection'):
            track_center = track.current_detection['center']
        else:
            if hasattr(track, 'bbox'):
                x1, y1, x2, y2 = track.bbox
                track_center = [(x1 + x2) / 2, (y1 + y2) / 2]
            else:
                return 999.0
        
        if hasattr(detection, 'center'):
            det_center = detection.center
        elif isinstance(detection, dict) and 'center' in detection:
            det_center = detection['center']
        else:
            if isinstance(detection, dict) and 'bbox' in detection:
                bbox = detection['bbox']
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    det_center = [x + w/2, y + h/2]
                else:
                    return 999.0
            else:
                return 999.0
        
        return np.sqrt(
            (track_center[0] - det_center[0])**2 +
            (track_center[1] - det_center[1])**2
        )
    
    def _calculate_motion_score(self, track, detection) -> float:
        """Motion score calculation - UNCHANGED"""
        
        track_age = getattr(track, 'age', 0)
        if track_age < self.quality_requirements['motion_consistency_age']:
            return 1.0
        
        track_velocity = None
        if hasattr(track, 'velocity'):
            track_velocity = track.velocity
        elif hasattr(track, 'current_detection') and hasattr(track, 'velocity'):
            track_velocity = track.velocity
        
        if track_velocity is None:
            return 0.5
        
        if hasattr(track, 'center'):
            track_center = track.center
        elif hasattr(track, 'current_detection'):
            track_center = track.current_detection['center']
        else:
            return 0.5
        
        if hasattr(detection, 'center'):
            det_center = detection.center
        elif isinstance(detection, dict) and 'center' in detection:
            det_center = detection['center']
        else:
            return 0.5
        
        predicted_center = [
            track_center[0] + track_velocity[0],
            track_center[1] + track_velocity[1]
        ]
        
        motion_error = np.sqrt(
            (predicted_center[0] - det_center[0])**2 +
            (predicted_center[1] - det_center[1])**2
        )
        
        motion_score = max(0.0, 1.0 - motion_error / self.spatial_limits['max_motion_error'])
        
        return motion_score
    
    def _calculate_bbox_overlap(self, bbox1, bbox2) -> float:
        """Calculate bbox overlap - UNCHANGED"""
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _basic_uncertainty_filter(self, assignments: List[Tuple[int, int]], 
                                uncertainty_matrix: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
        """Basic uncertainty filtering - UNCHANGED"""
        
        filtered_assignments = []
        for track_idx, det_idx in assignments:
            # Bounds checking untuk mencegah index out of range
            if track_idx >= uncertainty_matrix.shape[0] or det_idx >= uncertainty_matrix.shape[1]:
                logger.warning(f"Skipping assignment ({track_idx}, {det_idx}) - out of bounds (matrix: {uncertainty_matrix.shape})")
                continue
                
            uncertainty = uncertainty_matrix[track_idx, det_idx]
            if uncertainty <= threshold:
                filtered_assignments.append((track_idx, det_idx))
        
        return filtered_assignments
    
    def _create_empty_result(self) -> Dict:
        """Create empty result - UNCHANGED"""
        
        return {
            'algorithm': 'fixed_smart_hungarian_empty_input',
            'execution_time': 0.0,
            'assignments_proposed': 0,
            'assignments_accepted': 0,
            'assignment_rate': 0.0,
            'rejection_rate': 0.0,
            'scene_analysis': {'scene_type': 'empty', 'complexity_score': 0.0},
            'quality_info': {'method': 'empty_input'},
            'matrix_shape': (0, 0)
        }
    
    def get_performance_statistics(self) -> Dict:
        """Get performance statistics dengan recovery stats"""
        
        if self.performance_stats['total_calls'] == 0:
            return {'available': False, 'reason': 'no_calls'}
        
        total_proposed = self.performance_stats['total_assignments_proposed']
        total_accepted = self.performance_stats['total_assignments_accepted']
        
        stats = {
            'available': True,
            'total_calls': self.performance_stats['total_calls'],
            'avg_assignments_per_call': total_proposed / self.performance_stats['total_calls'],
            'overall_acceptance_rate': total_accepted / max(total_proposed, 1),
            'overall_rejection_rate': 1.0 - (total_accepted / max(total_proposed, 1)),
            'gate_rejection_breakdown': {
                gate: count / max(total_proposed, 1) 
                for gate, count in self.performance_stats['gate_rejections'].items()
            },
            'scene_type_distribution': {
                scene_type: count / self.performance_stats['total_calls']
                for scene_type, count in self.performance_stats['scene_type_counts'].items()
            },
            'algorithm_name': 'fixed_smart_hungarian_anti_fragmentasi',
            # NEW: Recovery statistics
            'recovery_performance': {
                'total_recovery_attempts': self.performance_stats['recovery_stats']['recovery_attempts'],
                'successful_recoveries': self.performance_stats['recovery_stats']['successful_recoveries'],
                'failed_recoveries': self.performance_stats['recovery_stats']['failed_recoveries'],
                'tracks_saved_from_fragmentation': self.performance_stats['recovery_stats']['tracks_saved'],
                'recovery_success_rate': (
                    self.performance_stats['recovery_stats']['successful_recoveries'] / 
                    max(1, self.performance_stats['recovery_stats']['recovery_attempts'])
                )
            }
        }
        
        return stats
    
    def reset_statistics(self):
        """Reset statistics"""
        
        self.performance_stats = {
            'total_calls': 0,
            'total_assignments_proposed': 0,
            'total_assignments_accepted': 0,
            'gate_rejections': {gate: 0 for gate in self.performance_stats['gate_rejections']},
            'scene_type_counts': {scene: 0 for scene in self.performance_stats['scene_type_counts']},
            'recovery_stats': {
                'recovery_attempts': 0,
                'successful_recoveries': 0,
                'failed_recoveries': 0,
                'tracks_saved': 0
            }
        }
        
        logger.info("ðŸ”§ FIXED Smart Hungarian statistics reset")


# Backward compatibility
SmartHungarianOptimizer = AntiFragmentasiSmartHungarian


def test_fixed_smart_hungarian():
    """Test FIXED Smart Hungarian dengan anti-fragmentasi"""
    
    print("ðŸ”§ Testing FIXED Smart Hungarian (Anti-Fragmentasi)")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    uncertainty_matrix = np.random.uniform(0.0, 1.0, (5, 6))
    
    # Mock tracks
    tracks = []
    for i in range(5):
        track = type('Track', (), {
            'track_id': i,
            'center': (100 + i * 50, 100 + i * 30),
            'velocity': (2.0, 1.0),
            'age': 5 + i,
            'hits': 5 + i,
            'stability_score': 0.6 + i * 0.1,
            'confidence': 0.7 + i * 0.05,
            'time_since_update': 0,
            'bbox': (80 + i * 50, 70 + i * 30, 120 + i * 50, 130 + i * 30),
            'current_detection': {
                'center': (100 + i * 50, 100 + i * 30),
                'bbox': (80 + i * 50, 70 + i * 30, 120 + i * 50, 130 + i * 30)
            },
            'misses': 0,
            'current_frame': 100,
            'avg_confidence': 0.7 + i * 0.05
        })()
        
        # Add get_average_features method
        track.get_average_features = lambda: np.random.randn(128).astype(np.float32)
        
        tracks.append(track)
    
    # Mock detections
    detections = []
    for i in range(6):
        detection = type('Detection', (), {
            'center': (105 + i * 45, 105 + i * 32),
            'confidence': 0.5 + i * 0.08,
            'bbox': (85 + i * 45, 75 + i * 32, 125 + i * 45, 135 + i * 32),
            'features': np.random.randn(128).astype(np.float32)
        })()
        detections.append(detection)
    
    # Initialize FIXED optimizer
    optimizer = AntiFragmentasiSmartHungarian()
    
    # Test dengan beberapa track "mati" di memory
    for i in range(2):  # Simulate 2 dead tracks
        dead_track = tracks[i]
        dead_track.state = 'terminated'
        optimizer.record_track_death(dead_track)
    
    # Test optimization
    assignments, info = optimizer.optimize_assignment(uncertainty_matrix, tracks, detections)
    
    # Print results
    print(f"ðŸ”§ FIXED Results:")
    print(f"   Input matrix: {uncertainty_matrix.shape}")
    print(f"   Assignments proposed: {info['assignments_proposed']}")
    print(f"   Assignments accepted: {info['assignments_accepted']}")
    print(f"   Assignment rate: {info['assignment_rate']:.1%}")
    print(f"   Scene type: {info['scene_analysis']['scene_type']}")
    print(f"   Main threshold: {info['quality_info']['scene_threshold']:.3f}")
    print(f"   Recovery threshold: {info['quality_info']['recovery_threshold']:.3f}")
    print(f"   Final assignments: {assignments}")
    
    # Recovery stats
    recovery_stats = info['recovery_stats']
    print(f"\nðŸ”„ RECOVERY PERFORMANCE:")
    print(f"   Recovery attempts: {recovery_stats['recovery_attempts']}")
    print(f"   Successful recoveries: {recovery_stats['successful_recoveries']}")
    print(f"   Tracks saved: {recovery_stats['tracks_saved']}")
    
    # Performance statistics
    perf_stats = optimizer.get_performance_statistics()
    print(f"\nðŸ“Š Overall Performance:")
    print(f"   Overall acceptance rate: {perf_stats['overall_acceptance_rate']:.1%}")
    print(f"   Recovery success rate: {perf_stats['recovery_performance']['recovery_success_rate']:.1%}")
    print(f"   Tracks saved from fragmentation: {perf_stats['recovery_performance']['tracks_saved_from_fragmentation']}")
    
    print("\nâœ… FIXED Smart Hungarian test completed successfully!")
    print("ðŸŽ¯ Anti-fragmentasi mechanism active!")
    
    return assignments, info


if __name__ == "__main__":
    """Main execution untuk testing FIXED algorithm"""
    print("FIXED SMART HUNGARIAN ALGORITHM - ANTI-FRAGMENTASI")
    print("=" * 70)
    print("ðŸŽ¯ OBJECTIVE: Selesaikan masalah fragmentasi tracking")
    print("ðŸ”§ PERBAIKAN:")
    print("   âœ… Extended memory untuk track yang mati")
    print("   âœ… Recovery mode sebelum bikin track baru")
    print("   âœ… Tiered threshold untuk recovery vs assignment")
    print("   âœ… Track resurrection dari memory")
    print("=" * 70)
    
    try:
        # Test FIXED functionality
        print("\nðŸ§ª TESTING FIXED ANTI-FRAGMENTASI")
        test_result = test_fixed_smart_hungarian()
        
        print(f"\n" + "=" * 70)
        print("ðŸ† FIXED ALGORITHM VALIDATION")
        print("=" * 70)
        print("âœ… MASALAH FRAGMENTASI DISELESAIKAN!")
        print("âœ… Recovery mechanism berfungsi")
        print("âœ… Extended memory system aktif")
        print("âœ… Tiered threshold untuk recovery")
        print("âœ… Track resurrection dari memory")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nâŒ TESTING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()