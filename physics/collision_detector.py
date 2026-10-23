"""
Collision Detector for LTE-TrackGuard
======================================

Deteksi tabrakan/ringsek menggunakan prinsip fisika dari Blueprint:

PRIMARY METRICS (Blueprint Section 2.4.5):
1. Bounding box overlap violation (IoU > 0.3)
2. Kinetic energy dissipation (>90% loss dalam 5 frames)
3. No gradual deceleration (sudden stop via acceleration variance)

ENHANCEMENT: Deformation-Based Collision Detection
- Deformation Detection: Deteksi collision berdasarkan perubahan bentuk fisik (deformasi)
- Prinsip: Mobil yang tabrakan akan berubah bentuk (AR/area berubah drastis)
- Mobil yang hanya berdekatan di traffic padat: bentuk tetap sama → bukan collision

Physics Principles:
- Deformation: Perubahan Aspect Ratio (AR) atau Area yang signifikan dari baseline
- Baseline Tracking: Simpan AR dan area baseline untuk setiap track (average dari history)
- Deformation = AR_change > 20% ATAU Area_change > 15% dari baseline
- Collision = IoU tinggi + Energy Loss tinggi + Deformasi terdeteksi

From Blueprint:
- Section 2.4.5: Behaviour 4 - Deteksi Tabrakan (IoU + Energy Dissipation)
- Section 2.6.3: Momentum Conservation (energy transfer to deformation)
- Section 2.6: Physics-Based Prediction (deformation analysis)
"""

import time
import numpy as np
import math
import logging
import os
from collections import deque
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from physics.base_detector import BaseDetector, BehaviourState

if TYPE_CHECKING:
    from physics.scene_analyzer import SceneAnalyzer
    from physics.velocity_field import VelocityField

from physics.physics_primitive_layer import PhysicsPrimitiveLayer

logger = logging.getLogger(__name__)


class CollisionDetector(BaseDetector):
    """
    Detector untuk tabrakan menggunakan overlap + energy dissipation
    
    Physics metrics:
    - IoU overlap (spatial violation)
    - Energy loss (kinetic energy dissipation)
    - Acceleration variance (sudden vs gradual)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize collision detector
        
        Args:
            config: Configuration dari PHYSICS_CONFIG['collision_detector']
        """
        super().__init__(config, "CollisionDetector")
        
        # Thresholds dari blueprint (Blueprint Section 2.4.5)
        self.iou_overlap_threshold = config.get('iou_overlap_threshold', 0.3)
        self.energy_loss_threshold = config.get('energy_loss_threshold', 0.9)  # Blueprint strict: 0.9 (90%)
        self.variance_acceleration = config.get('variance_acceleration', 5.0)  # Blueprint: 5.0
        self.energy_loss_frames_back = config.get('energy_loss_frames_back', 5)  # Blueprint: 5 frames
        self.variance_frames_required = config.get('variance_frames_required', 4)  # Reduced from 6
        
        # ============================================
        # OVERLAP DURATION TRACKING (NEW - untuk distinguish collision vs traffic jam)
        # Prinsip: Collision = TRANSIENT event (< 5 frames)
        #          Traffic Jam = PERSISTENT state (> 10 frames)
        # ============================================
        self.overlap_history = {}  # {(track_i_id, track_j_id): {'frames': [frame_ids], 'last_iou': float}}
        self.transient_overlap_max = config.get('transient_overlap_max', 5)  # Max 5 frames = collision
        self.persistent_overlap_min = config.get('persistent_overlap_min', 10)  # Min 10 frames = traffic jam
        self.overlap_history_cleanup_interval = 100  # Cleanup every 100 frames
        
        # Use higher persistence untuk avoid false alarms (Blueprint: W=3, τ=0.8)
        # REDUCED untuk deteksi lebih cepat: 1 frame (deteksi instan)
        self.persist_threshold = config.get('persist_threshold', 0.8)
        self.persist_window = config.get('persist_window', 1)  # REDUCED: 1 frame untuk deteksi instan
        
        # Impulse-Based Detection (NEW - untuk deteksi berbasis gaya tabrakan)
        self.impulse_threshold = config.get('impulse_threshold', 50.0)  # Minimum impulse magnitude (px²/frame)
        self.impulse_frames_back = config.get('impulse_frames_back', 2)  # Frames to look back for velocity change
        
        # Simultaneous Acceleration Spike Detection (NEW)
        self.acceleration_spike_threshold = config.get('acceleration_spike_threshold', 10.0)  # px/frame²
        self.acceleration_spike_frames_back = config.get('acceleration_spike_frames_back', 2)  # Frames to check
        
        # Force Magnitude Detection (NEW)
        self.force_threshold = config.get('force_threshold', 100.0)  # Minimum force magnitude (px²/frame²)
        
        # Deformation Detection (NEW - untuk deteksi mobil ringsek)
        self.ar_change_threshold = config.get('ar_change_threshold', 0.12)  # 12% perubahan AR (relaxed)
        self.area_change_threshold = config.get('area_change_threshold', 0.10)  # 10% perubahan area (relaxed)
        self.baseline_frames = config.get('baseline_frames', 10)  # Frames untuk baseline calculation
        # Multi-Tier Detection Strategy (untuk handle rear-end collision tanpa deformasi jelas)
        # Tier 1: High-Confidence (tidak perlu deformation)
        self.tier1_iou_threshold = config.get('tier1_iou_threshold', 0.5)
        self.tier1_energy_threshold = config.get('tier1_energy_threshold', 0.8)
        self.tier1_variance_threshold = config.get('tier1_variance_threshold', 5.0)
        self.tier1_rotation_spike_threshold = config.get('tier1_rotation_spike_threshold', 0.8)
        # Tier 1.5: Sparse Scene High-Confidence
        self.tier1_5_sparse_density_threshold = config.get('tier1_5_sparse_density_threshold', 1.0)
        self.tier1_5_iou_threshold = config.get('tier1_5_iou_threshold', 0.4)
        self.tier1_5_energy_threshold = config.get('tier1_5_energy_threshold', 0.7)
        self.tier1_5_rotation_spike_threshold = config.get('tier1_5_rotation_spike_threshold', 0.6)
        # Push Collision Detection (Rear-End) - NEW - PRIORITY TIER
        self.tier0_5_push_iou_threshold = config.get('tier0_5_push_iou_threshold', 0.25)
        self.tier0_5_push_energy_threshold = config.get('tier0_5_push_energy_threshold', 0.5)
        self.momentum_transfer_threshold = config.get('momentum_transfer_threshold', 20.0)
        self.relative_velocity_drop_threshold = config.get('relative_velocity_drop_threshold', 0.4)
        self.push_acceleration_threshold = config.get('push_acceleration_threshold', 3.0)
        self.velocity_direction_change_threshold = config.get('velocity_direction_change_threshold', 20.0)
        self.push_frames_back = config.get('push_frames_back', 2)
        # Traffic Jam Filter
        self.traffic_jam_velocity_threshold = config.get('traffic_jam_velocity_threshold', 2.0)
        self.traffic_jam_velocity_change_threshold = config.get('traffic_jam_velocity_change_threshold', 0.2)
        self.traffic_jam_energy_loss_threshold = config.get('traffic_jam_energy_loss_threshold', 0.2)
        self.traffic_jam_acceleration_variance_threshold = config.get('traffic_jam_acceleration_variance_threshold', 2.0)
        # Proximity Warning
        self.proximity_iou_threshold = config.get('proximity_iou_threshold', 0.1)
        self.proximity_velocity_threshold = config.get('proximity_velocity_threshold', 3.0)
        self.proximity_frames_back = config.get('proximity_frames_back', 5)
        # Tier 2: Deformation-Based (perlu deformation)
        self.tier2_iou_threshold = config.get('tier2_iou_threshold', 0.3)
        self.tier2_energy_threshold = config.get('tier2_energy_threshold', 0.9)
        # Tier 3: Medium-Confidence (fallback)
        self.tier3_iou_threshold = config.get('tier3_iou_threshold', 0.4)
        self.tier3_energy_threshold = config.get('tier3_energy_threshold', 0.7)
        self.tier3_variance_threshold = config.get('tier3_variance_threshold', 3.0)
        
        # Baseline tracking untuk setiap track (AR dan area baseline)
        # Format: {track_id: {'ar_baseline': float, 'area_baseline': float, 'baseline_frames': list}}
        self.track_baselines = {}
        
        # Collision pairs tracking dengan state machine
        # Format: {(track_id_1, track_id_2): {'state': 'monitoring'|'confirmed', 'first_detected': frame_id,
        #           'persist_count': int, 'passing_candidate': bool}}
        self.collision_pairs = {}

        # Passing vehicle filter constants
        # Kendaraan papasan/crossing: bbox overlap sesaat di intersection (perpendicular/opposing),
        # bukan hanya head-on. Dot product < +0.5 tangkap: opposing (-1), perpendicular (0), slight-cross (+0.3).
        # Tabrakan nyata: IoU bertahan >= _PASSING_CONFIRM_FRAMES frame berturut-turut.
        # PPL+KELR bypass hanya jika KELR sangat kuat (>0.9) DAN IoU besar (>0.3) — bukan sekadar braking biasa.
        self._PASSING_MIN_SPEED = 2.0      # px/frame — minimum speed to consider "moving"
        self._PASSING_DOT_THRESH = 0.5     # dot product unit vectors — <0.5 = crossing/opposing/perpendicular
        self._PASSING_CONFIRM_FRAMES = 5   # frame persistence needed to confirm passing_candidate as real collision
        self._PASSING_KELR_BYPASS = 0.9    # KELR value threshold for hard bypass (very strong impact evidence)
        self._PASSING_IOU_BYPASS = 0.3     # IoU threshold for hard bypass (significant physical overlap)
        # Post-contact physical evidence thresholds
        # "Selama belum ada bukti deformasi fisik atau kendaraan berhenti, itu bukan tabrakan"
        self._DECEL_RATIO = 0.5            # speed harus turun ke <50% dari speed_at_contact
        self._AR_CHANGE_SIGMA = 1.5        # ARS z-score persisten untuk confirm deformasi bbox
        self._POST_CONTACT_WINDOW = 15     # frame setelah kontak untuk tunggu evidence

        # Proximity warnings tracking (for visualization)
        # Format: {(track_id_1, track_id_2): {'track_i': int, 'track_j': int, 'bbox_i': list, 'bbox_j': list, 'iou': float, 'frame_id': int}}
        self.proximity_warnings = {}

        # Proximity monitoring for disappearance-based collision detection (HYBRID SOLUTION)
        # Format: {(track_id_1, track_id_2): {'frames': [frame_ids], 'max_iou': float, 'last_frame': int, 'class_i': str, 'class_j': str}}
        self.proximity_monitoring = {}
        self.proximity_monitoring_window = 10  # Monitor dalam 10 frame terakhir
        self.disappearance_collision_iou_threshold = 0.2  # Min IoU untuk consider disappearance collision

        # Track last seen frame (untuk detect sudden disappearance)
        # Format: {track_id: {'last_frame': int, 'confidence': float, 'bbox': list, 'class_name': str}}
        self.track_last_seen = {}

        # Evasive maneuver detection (NEW - untuk detect mobil yang tiba-tiba oleng)
        # Format: {track_id: {'velocities': deque, 'directions': deque, 'frames': deque}}
        self.track_motion_history = {}
        self.motion_history_window = 5  # Track 5 frame terakhir untuk hitung direction change

        # Evasive maneuver thresholds (STRICT - untuk filter normal driving behavior)
        # CRITICAL: These thresholds MUST be high enough to avoid false positives from normal turns/lane changes
        # RAISED significantly to distinguish normal turn vs extreme evasive maneuver
        # Minimum track quality for collision detection
        # Ghost/short-lived tracks (hits < 5) are common FP source in Smart Hungarian
        self.min_track_hits = config.get('min_track_hits', 5)

        # Configurable collision confidence threshold (default 70 = COLLISION DETECTED)
        self.collision_confidence_threshold = config.get('collision_confidence_threshold', 70)

        # Episode-level escalation gate.
        # A frame-level PPL warning remains yellow immediately. It is upgraded to
        # red only when the short interaction episode accumulates a strong ARS
        # shock while KELR remains below the saturated false-warning pattern.
        escalation_cfg = config.get('episode_escalation', {})
        self.episode_escalation_enabled = escalation_cfg.get('enabled', True)
        self.episode_escalation_ars_threshold = escalation_cfg.get('ars_threshold', 2.9)
        self.episode_escalation_kelr_max = escalation_cfg.get('kelr_max_threshold', 0.89)
        self.episode_escalation_gap = escalation_cfg.get('gap_frames', 30)
        self.episode_escalation_state = {}

        evasive_escalation_cfg = config.get('evasive_escalation', {})
        self.evasive_escalation_enabled = evasive_escalation_cfg.get('enabled', True)
        self.evasive_escalation_direction_threshold = evasive_escalation_cfg.get('direction_change_threshold', 135.0)
        self.evasive_escalation_iou_threshold = evasive_escalation_cfg.get('iou_threshold', 0.004)
        self.evasive_escalation_kelr_max = evasive_escalation_cfg.get('kelr_max_threshold', 0.95)

        self.evasive_angular_velocity_threshold = config.get('evasive_angular_velocity_threshold', 40.0)   # degree/frame - RAISED from 20.0 to 40.0 (hanya extreme evasive)
        self.evasive_lateral_accel_threshold = config.get('evasive_lateral_accel_threshold', 15.0)         # px/fr² - RAISED from 8.0 to 15.0 (hanya extreme evasive)
        self.evasive_min_speed = config.get('evasive_min_speed', 1.0)                                      # px/fr - Must be moving significantly
        
        # Normal turn filter thresholds (untuk filter belok normal dari evasive collision detection)
        # Jika angular velocity < ini, berarti belok normal (bukan evasive extreme)
        self.normal_turn_angular_velocity_max = config.get('normal_turn_angular_velocity_max', 35.0)  # degree/frame - maks untuk belok normal
        self.normal_turn_lateral_accel_max = config.get('normal_turn_lateral_accel_max', 12.0)        # px/fr² - maks untuk belok normal

        # Evasive collision IoU threshold (STRICT - must overlap significantly)
        self.evasive_collision_iou_threshold = config.get('evasive_collision_iou_threshold', 0.8)  # RAISED from 0.3 to 0.5

        # Evasive collision confidence threshold (LOWERED untuk detect lebih banyak collision yang valid)
        self.evasive_collision_confidence_threshold = config.get('evasive_collision_confidence_threshold', 55)

        # Recently disappeared tracks (untuk cross-reference dengan evasive maneuver)
        # Format: {track_id: {'disappeared_frame': int, 'last_bbox': list, 'class_name': str}}
        self.recently_disappeared = {}
        self.disappeared_retention_frames = 10  # Simpan info track yang hilang selama 10 frame

        # Sticky class: once a track_id is ever detected as truck/bus, always treat it
        # as truck/bus — prevents YOLO frame-to-frame class fluctuation from bypassing guards.
        # Format: {track_id: 'truck'|'bus'}
        self._track_sticky_class: dict = {}

        # Scene analyzer reference (will be set in detect() if available)
        self.scene_analyzer = None
        self.total_tracks_created_so_far = 0
        self.low_fragment_escalation_enabled = config.get('low_fragment_escalation_enabled', True)
        self.low_fragment_track_threshold = config.get('low_fragment_track_threshold', 100)
        self.low_fragment_active_track_threshold = config.get('low_fragment_active_track_threshold', 9)
        self.low_fragment_kelr_min = config.get('low_fragment_kelr_min', 0.50)

        # Global red-escalation scene-complexity guard. Critical/yellow evidence
        # is still emitted, but high-severity red is reserved for scenes where
        # active track load and cumulative fragmentation remain bounded.
        red_guard_cfg = config.get('red_scene_guard', {})
        self.red_scene_guard_enabled = red_guard_cfg.get('enabled', True)
        self.red_scene_guard_active_track_threshold = red_guard_cfg.get('active_tracks_max', 20)
        self.red_scene_guard_track_created_threshold = red_guard_cfg.get('tracks_created_max', 50)
        self.current_active_track_count = 0

        ppl_rescue_cfg = config.get('ppl_strong_rescue', {})
        self.ppl_strong_rescue_enabled = ppl_rescue_cfg.get('enabled', True)
        self.ppl_strong_rescue_frames = ppl_rescue_cfg.get('frames', 3)
        self.ppl_strong_rescue_iou_min = ppl_rescue_cfg.get('iou_min', 0.90)
        self.ppl_strong_rescue_min_kelr = ppl_rescue_cfg.get('min_kelr', 0.95)

        # ============================================
        # PHYSICS PRIMITIVE LAYER (PPL)
        # Tracker-agnostic, dimensionless collision oracle.
        # Sits above multi-tier heuristics — if PPL fires, skip all tiers.
        # ============================================
        self.ppl = PhysicsPrimitiveLayer(fps=30.0, config=config.get('ppl', {}))

        print(f"  IoU overlap threshold: {self.iou_overlap_threshold}")
        print(f"  Energy loss threshold: {self.energy_loss_threshold} ({self.energy_loss_threshold*100:.0f}%) - ADAPTIVE")
        print(f"  Acceleration variance threshold: {self.variance_acceleration}")
        print(f"  Energy loss frames back: {self.energy_loss_frames_back}")
        print(f"  Variance frames required: {self.variance_frames_required}")
        print(f"  Persistence: {self.persist_window} frames, threshold: {self.persist_threshold}")
        print(f"  Impulse threshold: {self.impulse_threshold} px²/frame")
        print(f"  Acceleration spike threshold: {self.acceleration_spike_threshold} px/frame²")
        print(f"  Force threshold: {self.force_threshold} px²/frame²")
        print(f"  Deformation detection: AR change > {self.ar_change_threshold*100:.0f}% OR Area change > {self.area_change_threshold*100:.0f}%")
        print(f"  Min track hits: {self.min_track_hits} (ghost track filter)")
        print(f"  Collision confidence threshold: {self.collision_confidence_threshold}%")
        print(
            "  Episode escalation gate: "
            f"enabled={self.episode_escalation_enabled}, "
            f"ARS>={self.episode_escalation_ars_threshold}, "
            f"KELR<={self.episode_escalation_kelr_max}"
        )
        print(
            "  Evasive escalation gate: "
            f"enabled={self.evasive_escalation_enabled}, "
            f"dir>={self.evasive_escalation_direction_threshold}, "
            f"IoU>={self.evasive_escalation_iou_threshold}, "
            f"KELR<={self.evasive_escalation_kelr_max}"
        )
        print(
            "  Low-fragment critical escalation: "
            f"enabled={self.low_fragment_escalation_enabled}, "
            f"tracks_created<={self.low_fragment_track_threshold}, "
            f"active_tracks<={self.low_fragment_active_track_threshold}, "
            f"KELR>={self.low_fragment_kelr_min}"
        )
        print(
            "  Global red scene guard: "
            f"enabled={self.red_scene_guard_enabled}, "
            f"active_tracks<={self.red_scene_guard_active_track_threshold}, "
            f"tracks_created<={self.red_scene_guard_track_created_threshold}"
        )
        print(
            "  PPL strong rescue gate: "
            f"enabled={self.ppl_strong_rescue_enabled}, "
            f"frames>={self.ppl_strong_rescue_frames}, "
            f"IoU>={self.ppl_strong_rescue_iou_min}, "
            f"minKELR>={self.ppl_strong_rescue_min_kelr}"
        )
    
    def update_fps(self, fps: float):
        """Update FPS untuk PhysicsPrimitiveLayer (panggil saat video metadata diketahui)."""
        self.ppl.update_fps(fps)
        
        # Scale windows based on FPS (default tuned for 30 FPS)
        fps_ratio = fps / 30.0
        if fps > 40:
            # Scale windows to maintain temporal consistency (e.g., 5 frames @ 30fps = 0.16s)
            self.energy_loss_frames_back = int(5 * fps_ratio)
            self.variance_frames_required = int(4 * fps_ratio)
            self.motion_history_window = int(5 * fps_ratio)
            self.proximity_frames_back = int(5 * fps_ratio)
            logger.warning(f"  ⏭️ FPS detected: {fps} | Rescaling physics windows: energy_loss_frames={self.energy_loss_frames_back}")

    def reset_ppl(self):
        """Reset PPL IoU history antar video."""
        self.ppl.reset()
        self.episode_escalation_state.clear()

    def update_scene_info(self, density: float, category: str) -> None:
        """Forward scene density ke PPL scalar log. Dipanggil dari track_manager per-frame."""
        self.ppl.update_scene_info(density, category)

    def _red_scene_guard_allows(self, tracks: Optional[List] = None) -> tuple:
        """Return whether a candidate may be promoted to red escalation."""
        if not self.red_scene_guard_enabled:
            active_tracks = len(tracks) if tracks is not None else int(getattr(self, 'current_active_track_count', 0) or 0)
            tracks_created = int(getattr(self, 'total_tracks_created_so_far', 0) or 0)
            return True, active_tracks, tracks_created

        active_tracks = len(tracks) if tracks is not None else int(getattr(self, 'current_active_track_count', 0) or 0)
        tracks_created = int(getattr(self, 'total_tracks_created_so_far', 0) or 0)
        allowed = (
            active_tracks <= self.red_scene_guard_active_track_threshold
            and tracks_created <= self.red_scene_guard_track_created_threshold
        )
        return allowed, active_tracks, tracks_created

    def _apply_global_red_scene_guard(self, confidence: float, tracks: Optional[List], frame_id: int,
                                      pair_desc: str, context: str) -> tuple:
        """Promote/demote red while preserving critical interaction evidence."""
        if confidence < 50:
            return confidence, False, False

        allowed, active_tracks, tracks_created = self._red_scene_guard_allows(tracks)
        if allowed:
            if confidence < self.collision_confidence_threshold:
                logger.warning(
                    f"[SCENE-GUARDED RED ESCALATION] Frame {frame_id}: {pair_desc} | "
                    f"context={context} | active_tracks={active_tracks} <= {self.red_scene_guard_active_track_threshold} | "
                    f"tracks_created={tracks_created} <= {self.red_scene_guard_track_created_threshold} | "
                    f"Critical interaction upgraded to red escalation"
                )
            return max(confidence, self.collision_confidence_threshold), True, False

        if confidence >= self.collision_confidence_threshold:
            logger.warning(
                f"[RED ESCALATION SUPPRESSED] Frame {frame_id}: {pair_desc} | "
                f"context={context} | active_tracks={active_tracks} > {self.red_scene_guard_active_track_threshold} "
                f"or tracks_created={tracks_created} > {self.red_scene_guard_track_created_threshold} | "
                f"Kept as critical interaction"
            )
            return min(confidence, self.collision_confidence_threshold - 1), False, True

        return confidence, False, False

    def detect(self, tracks: List, velocity_field: 'VelocityField',
               scene_analyzer: Optional['SceneAnalyzer'] = None) -> List[Dict]:
        """
        Override detect untuk pairwise collision detection dengan INSTANT detection
        
        Args:
            tracks: List of track objects
            velocity_field: VelocityField object
            scene_analyzer: Optional SceneAnalyzer untuk adaptive thresholds
            
        Returns:
            List of confirmed collision detections
        """
        detections = []

        # R2Q4 per-stage profiler. PPL time is accumulated only around
        # PhysicsPrimitiveLayer.check_pair(); all remaining detector work is
        # reported as alarm-decision/policy time.
        _profile_detect_start = time.perf_counter()
        self._profile_ppl_seconds = 0.0
        self.last_ppl_compute_time = 0.0
        self.last_alarm_decision_time = 0.0
        self.last_collision_detector_time = 0.0
        
        # Store scene analyzer reference
        self.scene_analyzer = scene_analyzer
        
        # Get current frame ID (from first track if available)
        current_frame = tracks[0].current_frame if tracks and hasattr(tracks[0], 'current_frame') else 0
        self.current_active_track_count = len(tracks) if tracks is not None else 0
        
        # Get adaptive threshold berdasarkan scene density
        adaptive_threshold = self._get_adaptive_threshold()
        
        # Update baselines untuk semua tracks (AR dan area baseline)
        self._update_track_baselines(tracks)
        
        # Update motion history untuk semua tracks (untuk normal turn filter di pairwise collision check)
        for track in tracks:
            # Update motion history for active AND recently-ghosted tracks
            is_valid = self._is_valid_track(track)
            is_recent_ghost = (hasattr(track, 'state') and track.state == 'ghost' and
                              hasattr(track, 'misses') and track.misses <= 3 and
                              hasattr(track, 'hits') and track.hits >= 3)
            if is_valid or is_recent_ghost:
                # Pakai smoothed velocity (5-frame window) untuk direction history
                # supaya angular velocity tidak collapse ke 0.0°/fr akibat jitter.
                # compute_velocity() tetap dipakai di tempat lain (KELR, CE, dll).
                velocity = velocity_field.compute_velocity_smooth(track, window=5)
                self._update_motion_history(track.track_id, velocity, current_frame)

        # ============================================
        # HYBRID SOLUTION: Track disappearance-based collision detection
        # Step 1: Update track_last_seen untuk semua tracks yang aktif
        # ============================================
        active_track_ids = set()
        for track in tracks:
            # For track_last_seen: accept BOTH active AND recently-ghosted tracks
            # Ghost tracks need to be registered so disappearance detection works
            # (ghost = ByteTrack just lost it, collision detector needs to see the transition)
            is_active = self._is_valid_track(track)
            is_recent_ghost = (hasattr(track, 'state') and track.state == 'ghost' and
                              hasattr(track, 'misses') and track.misses <= 3 and
                              hasattr(track, 'hits') and track.hits >= 3)

            if is_active:
                track_id = track.track_id
                active_track_ids.add(track_id)
                _cls_now = track.current_detection.get('class_name', 'unknown')
                # Sticky class: sekali terdeteksi truck/bus, selalu truck/bus
                if _cls_now.lower() in ('truck', 'bus'):
                    self._track_sticky_class[track_id] = _cls_now.lower()
                self.track_last_seen[track_id] = {
                    'last_frame': current_frame,
                    'confidence': track.current_detection.get('confidence', 0.0),
                    'bbox': track.current_detection.get('bbox', [0, 0, 0, 0]),
                    'class_name': self._track_sticky_class.get(track_id, _cls_now),
                    'hits': track.hits if hasattr(track, 'hits') else 0,
                }
            elif is_recent_ghost:
                # Don't add to active_track_ids — this makes it appear as "disappeared"
                # But ensure it was in track_last_seen from a previous frame
                track_id = track.track_id
                if track_id not in self.track_last_seen:
                    # First time seeing this ghost — register with last known data
                    self.track_last_seen[track_id] = {
                        'last_frame': current_frame - 1,  # Pretend last seen 1 frame ago
                        'confidence': track.current_detection.get('confidence', 0.0),
                        'bbox': track.current_detection.get('bbox', [0, 0, 0, 0]),
                        'class_name': track.current_detection.get('class_name', 'unknown')
                    }

        # Step 2: Check untuk disappeared tracks (track yang ada di last_seen tapi tidak di active)
        disappeared_tracks = set(self.track_last_seen.keys()) - active_track_ids

        # [DISABLED] Disappearance collision check dinonaktifkan sementara — eksperimen FP reduction
        # Re-enable: hapus baris `if False:` dan unindent blok di bawah
        for disappeared_id in disappeared_tracks if False else []:
            # Check jika track hilang dalam 1-2 frame terakhir (sudden disappearance)
            last_seen_frame = self.track_last_seen[disappeared_id]['last_frame']
            frames_since_disappearance = current_frame - last_seen_frame

            if frames_since_disappearance <= 2:  # Sudden disappearance (1-2 frame)
                # Add to recently_disappeared untuk evasive collision detection
                if disappeared_id not in self.recently_disappeared:
                    self.recently_disappeared[disappeared_id] = {
                        'disappeared_frame': current_frame,
                        'last_bbox': self.track_last_seen[disappeared_id]['bbox'],
                        'class_name': self.track_last_seen[disappeared_id]['class_name']
                    }

                # Check proximity monitoring untuk disappeared track
                # ENHANCED: Check multiple evidence sources:
                # 1. Evasive maneuver dari vehicle (other track) setelah motor hilang
                # 2. Evasive maneuver dari motor yang hilang sebelum hilang (dari motion history)
                # 3. Relax requirement: vehicle speed tinggi + proximity tinggi = collision (rear-end collision)
                disappeared_class = self.track_last_seen[disappeared_id]['class_name'].lower()
                is_motorcycle_disappeared = 'motorcycle' in disappeared_class

                # Scope: hanya kendaraan roda 4 (car/truck/bus).
                # Skip motorcycle, person, dan unknown — bukan target deteksi tabrakan.
                _4wheel_classes = ('car', 'truck', 'bus')
                if not any(c in disappeared_class for c in _4wheel_classes):
                    logger.debug(f"[DISAPPEAR SKIP] Track {disappeared_id} ({disappeared_class}) — not a 4-wheel vehicle, excluded from disappearance check")
                    continue

                logger.warning(f"🔍 DISAPPEARANCE COLLISION CHECK | Track {disappeared_id} ({disappeared_class}) disappeared at frame {current_frame}")
                
                for pair_key, proximity_data in list(self.proximity_monitoring.items()):
                    if disappeared_id in pair_key:
                        # Track yang hilang punya proximity history!
                        other_id = pair_key[0] if pair_key[1] == disappeared_id else pair_key[1]
                        
                        # Get other track info dari proximity_data (untuk logging)
                        other_class_from_proximity = proximity_data.get('class_j' if pair_key[0] == other_id else 'class_i', 'unknown')
                        max_iou = proximity_data.get('max_iou', 0.0)
                        last_proximity_frame = proximity_data.get('last_frame', current_frame)
                        
                        logger.warning(f"  📍 Found proximity history: Track {disappeared_id} <-> Track {other_id} | "
                                     f"Max IoU: {max_iou:.3f} | Last proximity frame: {last_proximity_frame}")

                        # Check jika other track masih aktif
                        if other_id in active_track_ids:
                            # Get other track
                            other_track = next((t for t in tracks if t.track_id == other_id), None)
                            if other_track is None:
                                continue
                            
                            # CRITICAL: Ambil class dari track object langsung, bukan dari proximity_data
                            # proximity_data mungkin menyimpan class yang salah/terdahulu
                            other_class = other_track.current_detection.get('class_name', 'unknown')
                            other_is_vehicle = self._is_vehicle(other_track)
                            
                            logger.warning(f"  🚗 Other track {other_id} ({other_class}) is active | Is vehicle: {other_is_vehicle}")
                            
                            # ============================================
                            # EVIDENCE 1: Check evasive maneuver dari vehicle (other track) setelah motor hilang
                            # ============================================
                            velocity = velocity_field.compute_velocity(other_track, dt=1.0)
                            
                            # Check evasive maneuver dari vehicle
                            is_evasive_vehicle, angular_vel, lateral_acc, dir_change, sudden_dec_pct = self._detect_evasive_maneuver(
                                other_id, velocity
                            )
                            
                            logger.warning(f"  🔄 Vehicle evasive check: is_evasive={is_evasive_vehicle}, "
                                         f"angular={angular_vel:.1f}°/fr, lateral={lateral_acc:.2f} px/fr², "
                                         f"sudden_dec_pct={sudden_dec_pct:.1f}%")
                            
                            # ============================================
                            # EVIDENCE 2: Check evasive maneuver dari motor yang hilang (sebelum hilang)
                            # ============================================
                            is_evasive_disappeared = False
                            angular_vel_disappeared = 0.0
                            lateral_acc_disappeared = 0.0
                            dir_change_disappeared = 0.0
                            
                            # Check motion history dari motor yang hilang (sebelum hilang)
                            # Motion history masih tersedia karena di-update untuk semua tracks termasuk yang akan hilang
                            if disappeared_id in self.track_motion_history:
                                history = self.track_motion_history[disappeared_id]
                                if len(history['velocities']) >= 3 and len(history['directions']) >= 3:
                                    # Get last velocity sebelum hilang (frame terakhir sebelum disappearance)
                                    last_velocity = history['velocities'][-1]
                                    
                                    # Check evasive maneuver menggunakan motion history
                                    # Note: _detect_evasive_maneuver menggunakan track_motion_history, jadi bisa digunakan
                                    is_evasive_disappeared, angular_vel_disappeared, lateral_acc_disappeared, dir_change_disappeared, _ = self._detect_evasive_maneuver(
                                        disappeared_id, last_velocity
                                    )
                                    
                                    logger.warning(f"  🔄 Disappeared track evasive check (before disappearance): is_evasive={is_evasive_disappeared}, "
                                                 f"angular={angular_vel_disappeared:.1f}°/fr, lateral={lateral_acc_disappeared:.2f} px/fr²")
                                else:
                                    logger.warning(f"  ⚠️ Disappeared track {disappeared_id} motion history insufficient: "
                                                 f"velocities={len(history.get('velocities', []))}, "
                                                 f"directions={len(history.get('directions', []))}")
                            else:
                                logger.warning(f"  ⚠️ Disappeared track {disappeared_id} has no motion history")
                            
                            # ============================================
                            # EVIDENCE 3: Relax requirement untuk rear-end collision
                            # Jika: motorcycle hilang + vehicle dalam proximity tinggi + vehicle speed cukup tinggi
                            # ============================================
                            other_speed = np.linalg.norm(velocity)
                            has_high_speed_vehicle = (other_is_vehicle and other_speed >= 2.0)  # Vehicle speed >= 2.0 px/fr
                            has_high_proximity = (max_iou >= self.disappearance_collision_iou_threshold)  # IoU >= 0.2
                            should_relax_requirement = (is_motorcycle_disappeared and has_high_speed_vehicle and has_high_proximity)
                            
                            logger.warning(f"  ⚡ Relax requirement check: motorcycle={is_motorcycle_disappeared}, "
                                         f"vehicle_speed={other_speed:.2f} px/fr, high_proximity={has_high_proximity} (IoU={max_iou:.3f}) | "
                                         f"Should relax: {should_relax_requirement}")
                            
                            # ============================================
                            # EVIDENCE 4: Sudden brake dari vehicle (sudden deceleration > 50%)
                            # Motor hilang + mobil tiba-tiba ngerem = collision indicator
                            # ============================================
                            logger.warning(f"  [DEBUG PROXIMITY] Computing sudden brake and energy dissipation for Track {other_id}")
                            logger.warning(f"  [DEBUG PROXIMITY] sudden_dec_pct from evasive check: {sudden_dec_pct:.1f}%")
                            
                            has_sudden_brake = (sudden_dec_pct > 50.0)
                            
                            logger.warning(f"  🛑 Sudden brake check: sudden_dec_pct={sudden_dec_pct:.1f}% | "
                                         f"Has sudden brake: {has_sudden_brake} (threshold: >50%)")
                            
                            # ============================================
                            # EVIDENCE 5: Energy dissipation dari vehicle (> 50%)
                            # Motor hilang + mobil kehilangan energi drastis = collision indicator
                            # ============================================
                            logger.warning(f"  [DEBUG PROXIMITY] Computing energy loss for Track {other_id}")
                            energy_loss_vehicle = self._compute_energy_loss_instant(
                                other_track, velocity_field, self.energy_loss_threshold
                            )
                            logger.warning(f"  [DEBUG PROXIMITY] energy_loss computed: {energy_loss_vehicle:.3f}")
                            
                            has_energy_dissipation = (energy_loss_vehicle > 0.5)
                            
                            logger.warning(f"  ⚡ Energy dissipation check: energy_loss={energy_loss_vehicle:.3f} | "
                                         f"Has energy dissipation: {has_energy_dissipation} (threshold: >0.5)")
                            
                            # ============================================
                            # FILTER: Skip ALL pairs involving person/pedestrian
                            # Person = bukan vehicle collision, SELALU skip
                            # Mencegah FP dari orang nunggu nyebrang / jalan di pinggir jalan
                            # ============================================
                            other_class_lower = other_class.lower()
                            disappeared_class_lower = disappeared_class.lower()
                            is_other_person = self._is_pedestrian_class(other_class_lower)
                            is_other_motorcycle = 'motorcycle' in other_class_lower
                            is_disappeared_person = self._is_pedestrian_class(disappeared_class_lower)
                            is_disappeared_motorcycle = 'motorcycle' in disappeared_class_lower

                            has_person = is_disappeared_person or is_other_person
                            is_person_moto = self._is_person_motorcycle_pair(disappeared_class_lower, other_class_lower)

                            if has_person and not is_person_moto:
                                logger.warning(f"  ⏭️ SKIP disappearance collision | Track {disappeared_id} ({disappeared_class}) + Track {other_id} ({other_class}) | "
                                             f"Person+vehicle pair - skipping (not person+motorcycle)")
                                continue
                            
                            # Filter 4: Skip jika other track adalah motorcycle dan bukan vehicle (untuk evasive detection)
                            # Kecuali jika disappeared track juga motorcycle (motorcycle-motorcycle collision)
                            if is_other_motorcycle and not other_is_vehicle and not is_motorcycle_disappeared:
                                logger.warning(f"  ⏭️ SKIP disappearance collision | Track {disappeared_id} ({disappeared_class}) + Track {other_id} (motorcycle) | "
                                             f"Other track is motorcycle but not treated as vehicle - skipping collision detection")
                                continue
                            
                            # ============================================
                            # FILTER: Velocity dan speed validation (sebelum trigger collision)
                            # Filter ini sama dengan yang di _check_evasive_collision untuk konsistensi
                            # ============================================
                            # Get aggressor speed (other_track yang melakukan evasive atau yang dekat dengan disappeared track)
                            aggressor_speed = other_speed
                            
                            # FILTER 1: Skip jika velocity = 0.0 (stationary/calculation error)
                            if aggressor_speed == 0.0:
                                logger.warning(f"  ⏭️ SKIP disappearance collision | Track {other_id} | "
                                             f"Velocity = 0.0 (calculation failed or stationary) | "
                                             f"Skipping collision with Track {disappeared_id} - "
                                             f"likely static overlap or bbox jitter")
                                continue  # Skip to next disappeared track
                            
                            # (Person filters removed — person pairs already skipped entirely above)
                            
                            # ============================================
                            # DECISION: Trigger collision jika ada salah satu evidence
                            # ============================================
                            # Evidence combinations untuk collision:
                            # 1. Evasive maneuver dari vehicle (existing)
                            # 2. Evasive maneuver dari motor sebelum hilang (existing)
                            # 3. Relax requirement untuk rear-end collision (existing)
                            # 4. NEW: Sudden brake (>50%) + Energy dissipation (>50%) = strong collision evidence
                            # 5. NEW: Sudden brake (>50%) + High proximity (IoU > 0.2) = collision evidence
                            # 6. NEW: Energy dissipation (>50%) + High proximity (IoU > 0.2) = collision evidence
                            should_trigger_collision = (
                                is_evasive_vehicle or  # Vehicle melakukan evasive maneuver setelah motor hilang
                                (is_evasive_disappeared and is_motorcycle_disappeared) or  # Motor melakukan evasive maneuver sebelum hilang
                                should_relax_requirement or  # Relax requirement untuk rear-end collision
                                (has_sudden_brake and has_energy_dissipation) or  # NEW: Sudden brake + Energy dissipation
                                (has_sudden_brake and has_high_proximity) or  # NEW: Sudden brake + High proximity
                                (has_energy_dissipation and has_high_proximity)  # NEW: Energy dissipation + High proximity
                            )
                            
                            if should_trigger_collision:
                                if is_evasive_vehicle:
                                    logger.warning(f"  ✅ TRIGGER: Vehicle evasive maneuver detected")
                                if is_evasive_disappeared:
                                    logger.warning(f"  ✅ TRIGGER: Disappeared motorcycle had evasive maneuver")
                                if should_relax_requirement:
                                    logger.warning(f"  ✅ TRIGGER: Relax requirement met (rear-end collision likely)")
                                if has_sudden_brake and has_energy_dissipation:
                                    logger.warning(f"  ✅ TRIGGER: Sudden brake ({sudden_dec_pct:.1f}%) + Energy dissipation ({energy_loss_vehicle:.3f})")
                                if has_sudden_brake and has_high_proximity:
                                    logger.warning(f"  ✅ TRIGGER: Sudden brake ({sudden_dec_pct:.1f}%) + High proximity (IoU={max_iou:.3f})")
                                if has_energy_dissipation and has_high_proximity:
                                    logger.warning(f"  ✅ TRIGGER: Energy dissipation ({energy_loss_vehicle:.3f}) + High proximity (IoU={max_iou:.3f})")
                                
                                collision_generated = self._generate_disappearance_collision(
                                    disappeared_id, other_id, current_frame,
                                    proximity_data, tracks, velocity_field, detections
                                )

                                # Only log if collision was actually generated (not filtered)
                                if collision_generated:
                                    logger.warning(f"🔥 DISAPPEARANCE COLLISION DETECTED | "
                                                 f"Track {disappeared_id} ({disappeared_class}) disappeared after proximity with Track {other_id} ({other_class}) | "
                                                 f"Frame {current_frame} | Max IoU: {max_iou:.3f} | "
                                                 f"Evidence: vehicle_evasive={is_evasive_vehicle}, "
                                                 f"disappeared_evasive={is_evasive_disappeared}, "
                                                 f"relax_requirement={should_relax_requirement}, "
                                                 f"sudden_brake={has_sudden_brake} ({sudden_dec_pct:.1f}%), "
                                                 f"energy_dissipation={has_energy_dissipation} ({energy_loss_vehicle:.3f})")
                            else:
                                logger.warning(f"  ⏭️ SKIP disappearance collision | Track {disappeared_id} + {other_id} | "
                                             f"No sufficient evidence: vehicle_evasive={is_evasive_vehicle}, "
                                             f"disappeared_evasive={is_evasive_disappeared}, "
                                             f"relax_requirement={should_relax_requirement}, "
                                             f"sudden_brake={has_sudden_brake} ({sudden_dec_pct:.1f}%), "
                                             f"energy_dissipation={has_energy_dissipation} ({energy_loss_vehicle:.3f})")
                
                # ============================================
                # FALLBACK: Jika vehicle hilang tanpa proximity history, check semua vehicle aktif
                # untuk mencari kandidat yang mungkin terkait (rear-end collision, sudden disappearance)
                # EXTENDED: Tidak hanya motorcycle, tapi SEMUA vehicle (car, truck, bus, motorcycle)
                # ============================================
                # Check apakah disappeared track adalah vehicle (bukan person)
                is_vehicle_disappeared = any(vehicle_type in disappeared_class for vehicle_type in ['car', 'motorcycle', 'truck', 'bus'])

                if is_vehicle_disappeared:
                    has_proximity_history = any(disappeared_id in pair_key for pair_key in self.proximity_monitoring.keys())
                    logger.warning(f"  [DEBUG] Vehicle {disappeared_id} ({disappeared_class}) disappeared | Has proximity history: {has_proximity_history} | "
                                 f"Proximity monitoring keys: {list(self.proximity_monitoring.keys())}")

                    if not has_proximity_history:
                        # Guard: disappeared track harus cukup tua (hits >= 10 ≈ 1/3 detik @ 30fps)
                        # Track yang sangat baru (hits < 10) lebih mungkin oklusi atau deteksi noise
                        # daripada kendaraan yang benar-benar ditabrak.
                        disappeared_hits = self.track_last_seen[disappeared_id].get('hits', 0)
                        if disappeared_hits < 10:
                            logger.warning(f"  ⏭️ SKIP FALLBACK | Track {disappeared_id} terlalu baru "
                                         f"(hits={disappeared_hits} < 10) — likely occlusion/noise, bukan collision victim")
                            continue

                        logger.warning(f"  🔍 FALLBACK CHECK: No proximity history found, checking all active vehicles...")
                        logger.warning(f"  [DEBUG FALLBACK] Active tracks count: {len(tracks)}")

                        # Get last bbox dari vehicle yang hilang
                        last_bbox = self.track_last_seen[disappeared_id]['bbox']
                        logger.warning(f"  [DEBUG FALLBACK] Last bbox of disappeared vehicle {disappeared_id} ({disappeared_class}): {last_bbox}")
                        
                        # Check semua active vehicle tracks
                        vehicle_count = 0
                        for track in tracks:
                            if not self._is_vehicle(track):
                                continue
                            
                            vehicle_count += 1
                            other_id = track.track_id
                            if other_id == disappeared_id:
                                continue
                            
                            # Compute IoU dengan last bbox dari vehicle yang hilang
                            track_bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
                            iou = self._compute_iou(last_bbox, track_bbox)
                            
                            logger.warning(f"  [DEBUG FALLBACK] Checking vehicle Track {other_id} | IoU: {iou:.3f} | Bbox: {track_bbox}")
                            
                            # Check jika vehicle dalam jarak yang reasonable (IoU > 0.1 atau distance < threshold)
                            # Untuk rear-end collision, vehicle mungkin tidak overlap langsung tapi dalam jarak dekat
                            if iou > 0.1:  # Lower threshold untuk fallback check
                                logger.warning(f"  📍 FALLBACK: Found nearby vehicle Track {other_id} | IoU with last bbox: {iou:.3f}")
                                
                                # Get vehicle velocity
                                velocity = velocity_field.compute_velocity(track, dt=1.0)
                                other_speed = np.linalg.norm(velocity)
                                
                                logger.warning(f"  🚗 FALLBACK: Vehicle Track {other_id} | Speed: {other_speed:.2f} px/fr")
                                
                                # Check evasive maneuver dari vehicle
                                is_evasive_vehicle, angular_vel, lateral_acc, dir_change, sudden_dec_pct = self._detect_evasive_maneuver(
                                    other_id, velocity
                                )
                                
                                logger.warning(f"  🔄 FALLBACK Vehicle evasive check: Track {other_id} | "
                                             f"is_evasive={is_evasive_vehicle}, angular={angular_vel:.1f}°/fr, "
                                             f"lateral={lateral_acc:.2f} px/fr², speed={other_speed:.2f} px/fr, "
                                             f"sudden_dec_pct={sudden_dec_pct:.1f}%")
                                
                                # Check evasive maneuver dari vehicle yang hilang (sebelum hilang)
                                is_evasive_disappeared = False
                                if disappeared_id in self.track_motion_history:
                                    history = self.track_motion_history[disappeared_id]
                                    if len(history['velocities']) >= 3 and len(history['directions']) >= 3:
                                        last_velocity = history['velocities'][-1]
                                        is_evasive_disappeared, angular_vel_disappeared, lateral_acc_disappeared, dir_change_disappeared, _ = self._detect_evasive_maneuver(
                                            disappeared_id, last_velocity
                                        )
                                        
                                        logger.warning(f"  🔄 FALLBACK Disappeared track evasive check: is_evasive={is_evasive_disappeared}, "
                                                     f"angular={angular_vel_disappeared:.1f}°/fr, lateral={lateral_acc_disappeared:.2f} px/fr²")
                                
                                # ============================================
                                # FALLBACK EVIDENCE: Sudden brake dan energy dissipation
                                # ============================================
                                logger.warning(f"  [DEBUG FALLBACK] Computing sudden brake and energy dissipation for Track {other_id}")
                                
                                has_sudden_brake_fallback = (sudden_dec_pct > 50.0)
                                
                                logger.warning(f"  [DEBUG FALLBACK] sudden_dec_pct={sudden_dec_pct:.1f}%, has_sudden_brake={has_sudden_brake_fallback}")
                                
                                energy_loss_vehicle_fallback = self._compute_energy_loss_instant(
                                    track, velocity_field, self.energy_loss_threshold
                                )
                                has_energy_dissipation_fallback = (energy_loss_vehicle_fallback > 0.5)
                                
                                logger.warning(f"  [DEBUG FALLBACK] energy_loss={energy_loss_vehicle_fallback:.3f}, has_energy_dissipation={has_energy_dissipation_fallback}")
                                
                                has_high_proximity_fallback = (iou >= self.disappearance_collision_iou_threshold)  # IoU >= 0.2
                                
                                logger.warning(f"  🛑 FALLBACK Sudden brake check: sudden_dec_pct={sudden_dec_pct:.1f}% | "
                                             f"Has sudden brake: {has_sudden_brake_fallback} (threshold: >50%)")
                                logger.warning(f"  ⚡ FALLBACK Energy dissipation check: energy_loss={energy_loss_vehicle_fallback:.3f} | "
                                             f"Has energy dissipation: {has_energy_dissipation_fallback} (threshold: >0.5)")
                                logger.warning(f"  📍 FALLBACK High proximity check: IoU={iou:.3f} | "
                                             f"Has high proximity: {has_high_proximity_fallback} (threshold: >={self.disappearance_collision_iou_threshold})")
                                
                                # Decision: Trigger jika ada bukti fisik nyata.
                                # REMOVED: (other_speed >= 2.0 and iou > 0.1) — tidak ada dasar fisik,
                                # "kendaraan bergerak dekat track yang hilang" adalah oklusi, bukan tabrakan.
                                # Real collision memerlukan: evasive maneuver, pengereman mendadak,
                                # atau dissipasi energi — bukan sekadar kedekatan + kecepatan.
                                should_trigger = (
                                    is_evasive_vehicle or
                                    is_evasive_disappeared or
                                    (has_sudden_brake_fallback and has_energy_dissipation_fallback) or
                                    (has_sudden_brake_fallback and has_high_proximity_fallback) or
                                    (has_energy_dissipation_fallback and has_high_proximity_fallback)
                                )
                                
                                if should_trigger:
                                    trigger_reasons = []
                                    if is_evasive_vehicle:
                                        trigger_reasons.append("vehicle_evasive")
                                    if is_evasive_disappeared:
                                        trigger_reasons.append("disappeared_evasive")
                                    if has_sudden_brake_fallback and has_energy_dissipation_fallback:
                                        trigger_reasons.append(f"sudden_brake({sudden_dec_pct:.1f}%)+energy({energy_loss_vehicle_fallback:.3f})")
                                    if has_sudden_brake_fallback and has_high_proximity_fallback:
                                        trigger_reasons.append(f"sudden_brake({sudden_dec_pct:.1f}%)+proximity(IoU={iou:.3f})")
                                    if has_energy_dissipation_fallback and has_high_proximity_fallback:
                                        trigger_reasons.append(f"energy({energy_loss_vehicle_fallback:.3f})+proximity(IoU={iou:.3f})")
                                    
                                    logger.warning(f"  ✅ FALLBACK TRIGGER: Track {other_id} | "
                                                 f"Reasons: {', '.join(trigger_reasons)}")
                                    
                                    # Create proximity_data for fallback
                                    disappeared_last_frame = self.track_last_seen[disappeared_id]['last_frame']
                                    fallback_proximity_data = {
                                        'frames': [disappeared_last_frame],
                                        'max_iou': iou,
                                        'last_frame': disappeared_last_frame,
                                        'class_i': self.track_last_seen[disappeared_id]['class_name'],
                                        'class_j': track.current_detection.get('class_name', 'unknown')
                                    }
                                    
                                    collision_generated = self._generate_disappearance_collision(
                                        disappeared_id, other_id, current_frame,
                                        fallback_proximity_data, tracks, velocity_field, detections
                                    )
                                    
                                    if collision_generated:
                                        logger.warning(f"🔥 FALLBACK DISAPPEARANCE COLLISION DETECTED | "
                                                     f"Track {disappeared_id} ({disappeared_class}) disappeared near Track {other_id} | "
                                                     f"Frame {current_frame} | IoU: {iou:.3f}")

        # Cleanup disappeared tracks yang sudah lama (> 10 frames)
        for disappeared_id in list(disappeared_tracks):
            if current_frame - self.track_last_seen[disappeared_id]['last_frame'] > 10:
                del self.track_last_seen[disappeared_id]

        # Pairwise collision check (includes recently-ghosted tracks for ByteTrack hybrid)
        def _is_collision_eligible(t):
            if self._is_valid_track(t):
                return True
            # Allow recently-ghosted tracks (ByteTrack lost tracks)
            return (hasattr(t, 'state') and t.state == 'ghost' and
                    hasattr(t, 'misses') and t.misses <= 3 and
                    hasattr(t, 'hits') and t.hits >= 3)

        for i, track_i in enumerate(tracks):
            if not _is_collision_eligible(track_i):
                continue

            for j, track_j in enumerate(tracks):
                if i >= j:  # Avoid duplicate pairs
                    continue

                if not _is_collision_eligible(track_j):
                    continue
                
                # Check collision between track_i and track_j (INSTANT detection)
                collision_result, debug_info = self._check_pairwise_collision(
                    track_i, track_j, velocity_field, adaptive_threshold
                )
                
                # Handle proximity warning (return value bisa 'proximity' untuk warning)
                if collision_result == 'proximity':
                    # Proximity warning: objek mendekat, belum collision
                    # Store untuk visualization di main.py
                    pair_key = tuple(sorted([track_i.track_id, track_j.track_id]))
                    iou = debug_info.get('iou', 0.0)

                    self.proximity_warnings[pair_key] = {
                        'track_i': track_i.track_id,
                        'track_j': track_j.track_id,
                        'bbox_i': track_i.current_detection['bbox'],
                        'bbox_j': track_j.current_detection['bbox'],
                        'iou': iou,
                        'frame_id': current_frame
                    }

                    # ============================================
                    # HYBRID SOLUTION: Update proximity monitoring
                    # Monitor proximity untuk detect disappearance-based collision
                    # ============================================
                    if iou >= self.disappearance_collision_iou_threshold:
                        class_i = track_i.current_detection.get('class_name', 'unknown')
                        class_j = track_j.current_detection.get('class_name', 'unknown')

                        if pair_key not in self.proximity_monitoring:
                            self.proximity_monitoring[pair_key] = {
                                'frames': [current_frame],
                                'max_iou': iou,
                                'last_frame': current_frame,
                                'class_i': class_i,
                                'class_j': class_j
                            }
                        else:
                            # Update existing proximity monitoring
                            self.proximity_monitoring[pair_key]['frames'].append(current_frame)
                            self.proximity_monitoring[pair_key]['max_iou'] = max(
                                self.proximity_monitoring[pair_key]['max_iou'], iou
                            )
                            self.proximity_monitoring[pair_key]['last_frame'] = current_frame

                            # Keep only recent frames (sliding window)
                            self.proximity_monitoring[pair_key]['frames'] = [
                                f for f in self.proximity_monitoring[pair_key]['frames']
                                if current_frame - f <= self.proximity_monitoring_window
                            ]

                    # Skip collision detection untuk proximity warning
                    continue
                
                collision_detected = collision_result
                
                # Log every check for debugging
                detection_mode = debug_info.get('detection_mode', 'normal')
                
                # Log berdasarkan tier detection
                tier = debug_info.get('tier', 0)
                detection_tier = debug_info.get('detection_tier', '')
                if detection_tier == 'ppl':
                    # PhysicsPrimitiveLayer override — highest confidence
                    logger.warning(
                        f"[COLLISION PPL] Frame {current_frame}: "
                        f"Track {track_i.track_id} <-> Track {track_j.track_id} | "
                        f"PHYSICS PRIMITIVE LAYER | "
                        f"Path: {debug_info.get('ppl_path', '?')} | "
                        f"Strict: {debug_info.get('strict_scene', False)} | "
                        f"Primitives: {debug_info.get('ppl_primitives', '?')} | "
                        f"IoU: {debug_info.get('iou', 0):.3f} | "
                        f"Result: COLLISION CONFIRMED (physics-derived, tracker-agnostic)"
                    )
                elif detection_tier == 'post_collision':
                    # Post-collision static detection (ByteTrack hybrid)
                    logger.warning(f"[COLLISION POST-COLLISION] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                 f"POST-COLLISION STATIC DETECTION | "
                                 f"IoU: {debug_info.get('iou', 0):.3f} | "
                                 f"Past Speed: i={debug_info.get('variance_i', 0):.1f}, j={debug_info.get('variance_j', 0):.1f} px/fr | "
                                 f"Energy Loss (synthetic): {debug_info.get('energy_loss_i', 0):.3f}, {debug_info.get('energy_loss_j', 0):.3f} | "
                                 f"Result: COLLISION DETECTED (post-collision static)")
                elif tier == 0.5:
                    # Tier 0.5: Push Collision Priority
                    energy_loss_i = debug_info.get('energy_loss_i', 0.0)
                    energy_loss_j = debug_info.get('energy_loss_j', 0.0)
                    momentum_transfer_i = debug_info.get('momentum_transfer_i', 0.0)
                    momentum_transfer_j = debug_info.get('momentum_transfer_j', 0.0)
                    relative_velocity_drop = debug_info.get('relative_velocity_drop', 0.0) * 100
                    push_acceleration_i = debug_info.get('push_acceleration_i', 0.0)
                    push_acceleration_j = debug_info.get('push_acceleration_j', 0.0)
                    velocity_mismatch = debug_info.get('velocity_mismatch', False)
                    is_approaching = debug_info.get('is_approaching', False)
                    v_i_speed = debug_info.get('v_i_speed', 0.0)
                    v_j_speed = debug_info.get('v_j_speed', 0.0)
                    deformation_i = debug_info.get('deformation_i', False)
                    deformation_j = debug_info.get('deformation_j', False)
                    has_rear_end_evidence = debug_info.get('has_rear_end_evidence', False)
                    logger.warning(f"[COLLISION TIER0.5] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                 f"TIER 0.5: PUSH COLLISION PRIORITY (rear-end) | "
                                 f"IoU: {debug_info.get('iou', 0):.3f} | "
                                 f"Energy Loss: {energy_loss_i:.3f}, {energy_loss_j:.3f} | "
                                 f"Momentum Transfer: {momentum_transfer_i:.1f}, {momentum_transfer_j:.1f} | "
                                 f"Relative v Drop: {relative_velocity_drop:.0f}% | "
                                 f"Push Acc: {push_acceleration_i:.2f}, {push_acceleration_j:.2f} | "
                                 f"Velocity Mismatch: {velocity_mismatch} (v_i: {v_i_speed:.2f}, v_j: {v_j_speed:.2f}) | "
                                 f"Approaching: {is_approaching} | "
                                 f"Deformation: {deformation_i}, {deformation_j} | "
                                 f"Rear-end Evidence: {has_rear_end_evidence} | "
                                 f"Result: COLLISION DETECTED (push collision - rear-end priority)")
                elif tier == 1.5:
                    # Tier 1.5: Sparse Scene High-Confidence
                    energy_loss_i = debug_info.get('energy_loss_i', 0.0)
                    energy_loss_j = debug_info.get('energy_loss_j', 0.0)
                    rotation_spike_i = debug_info.get('rotation_spike_i', 0.0)
                    rotation_spike_j = debug_info.get('rotation_spike_j', 0.0)
                    logger.warning(f"[COLLISION TIER1.5] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                 f"TIER 1.5: SPARSE SCENE HIGH-CONFIDENCE | "
                                 f"IoU: {debug_info.get('iou', 0):.3f} | "
                                 f"Energy Loss: {energy_loss_i:.3f}, {energy_loss_j:.3f} | "
                                 f"Rotation Spike: {rotation_spike_i:.3f}, {rotation_spike_j:.3f} | "
                                 f"Result: COLLISION DETECTED (sparse scene, rotation spike)")
                elif tier == 1:
                    # Tier 1: High-Confidence
                    energy_loss_i = debug_info.get('energy_loss_i', 0.0)
                    energy_loss_j = debug_info.get('energy_loss_j', 0.0)
                    deformation_i = debug_info.get('deformation_i', False)
                    deformation_j = debug_info.get('deformation_j', False)
                    velocity_mismatch = debug_info.get('velocity_mismatch', False)
                    is_approaching = debug_info.get('is_approaching', False)
                    relative_velocity_drop = debug_info.get('relative_velocity_drop', 0.0) * 100
                    logger.warning(f"[COLLISION TIER1] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                 f"TIER 1: HIGH-CONFIDENCE | "
                                 f"IoU: {debug_info.get('iou', 0):.3f} | "
                                 f"Energy Loss: {energy_loss_i:.3f}, {energy_loss_j:.3f} | "
                                 f"Deformation: {deformation_i}, {deformation_j} | "
                                 f"Rear-end Evidence: Mismatch={velocity_mismatch}, Approaching={is_approaching}, Rel v Drop={relative_velocity_drop:.0f}% | "
                                 f"Result: COLLISION DETECTED (deformation or rear-end evidence)")
                elif tier == 2:
                    # Tier 2: Deformation-Based
                    ar_change_i = debug_info.get('ar_change_i', 0.0)
                    ar_change_j = debug_info.get('ar_change_j', 0.0)
                    area_change_i = debug_info.get('area_change_i', 0.0)
                    area_change_j = debug_info.get('area_change_j', 0.0)
                    logger.warning(f"[COLLISION TIER2] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                 f"TIER 2: DEFORMATION-BASED | "
                                 f"IoU: {debug_info.get('iou', 0):.3f} | "
                                 f"AR Change: {ar_change_i*100:.1f}%, {ar_change_j*100:.1f}% | "
                                 f"Area Change: {area_change_i*100:.1f}%, {area_change_j*100:.1f}% | "
                                 f"Result: COLLISION DETECTED (deformation-based)")
                elif tier == 3:
                    # Tier 3: Medium-Confidence
                    energy_loss_i = debug_info.get('energy_loss_i', 0.0)
                    energy_loss_j = debug_info.get('energy_loss_j', 0.0)
                    deformation_i = debug_info.get('deformation_i', False)
                    deformation_j = debug_info.get('deformation_j', False)
                    velocity_mismatch = debug_info.get('velocity_mismatch', False)
                    is_approaching = debug_info.get('is_approaching', False)
                    relative_velocity_drop = debug_info.get('relative_velocity_drop', 0.0) * 100
                    logger.warning(f"[COLLISION TIER3] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                 f"TIER 3: MEDIUM-CONFIDENCE (fallback) | "
                                 f"IoU: {debug_info.get('iou', 0):.3f} | "
                                 f"Energy Loss: {energy_loss_i:.3f}, {energy_loss_j:.3f} | "
                                 f"Deformation: {deformation_i}, {deformation_j} | "
                                 f"Rear-end Evidence: Mismatch={velocity_mismatch}, Approaching={is_approaching}, Rel v Drop={relative_velocity_drop:.0f}% | "
                                 f"Result: COLLISION DETECTED (deformation or rear-end evidence)")
                # Log normal checks (only if IoU > 0.1 to reduce spam)
                elif debug_info.get('iou', 0) > 0.1:
                    # Enhanced logging untuk kasus IoU tinggi tapi tidak terdeteksi
                    deformation_i = debug_info.get('deformation_i', False)
                    deformation_j = debug_info.get('deformation_j', False)
                    velocity_mismatch = debug_info.get('velocity_mismatch', False)
                    is_approaching = debug_info.get('is_approaching', False)
                    relative_velocity_drop = debug_info.get('relative_velocity_drop', 0.0) * 100
                    has_rear_end_evidence = debug_info.get('has_rear_end_evidence', False)
                    
                    if not collision_detected and debug_info.get('iou', 0) > 0.3:
                        # IoU tinggi tapi tidak terdeteksi - log reason
                        reason = []
                        if not deformation_i and not deformation_j:
                            reason.append("No deformation")
                        if not has_rear_end_evidence:
                            reason.append("No rear-end evidence")
                        if debug_info.get('is_traffic_jam', False):
                            reason.append("Traffic jam")
                        reason_str = " | ".join(reason) if reason else "Unknown"
                        
                        logger.warning(f"[COLLISION DEBUG] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                     f"IoU: {debug_info['iou']:.3f} (threshold: {self.iou_overlap_threshold}) | "
                                     f"Energy Loss: {debug_info['energy_loss_i']:.3f}, {debug_info['energy_loss_j']:.3f} | "
                                     f"Deformation: {deformation_i}, {deformation_j} | "
                                     f"Rear-end Evidence: {has_rear_end_evidence} (Mismatch={velocity_mismatch}, Approaching={is_approaching}, Rel v Drop={relative_velocity_drop:.0f}%) | "
                                     f"Result: NO COLLISION - Reason: {reason_str}")
                    else:
                        logger.warning(f"[COLLISION DEBUG] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                     f"IoU: {debug_info['iou']:.3f} (threshold: {self.iou_overlap_threshold}) | "
                                     f"Energy Loss: {debug_info['energy_loss_i']:.3f}, {debug_info['energy_loss_j']:.3f} "
                                     f"(threshold: {adaptive_threshold:.3f}) | "
                                     f"Variance: {debug_info['variance_i']:.3f}, {debug_info['variance_j']:.3f} "
                                     f"(threshold: {self.variance_acceleration}) | "
                                     f"Result: {'COLLISION DETECTED' if collision_detected else 'NO COLLISION'}")
                
                if collision_detected:
                    # Update state machine untuk persistence
                    pair_key = tuple(sorted([track_i.track_id, track_j.track_id]))

                    # ── Passing Vehicle Filter ────────────────────────────────────
                    # Kendaraan papasan/crossing di intersection: bbox overlap sesaat (1-3 frame)
                    # bukan tabrakan nyata. Tangkap dengan: arah crossing/perpendicular/opposing
                    # + keduanya masih kencang + IoU tidak signifikan.
                    # Bypass keras HANYA jika KELR sangat kuat (>0.9) DAN IoU besar (>0.3)
                    # — ini tabrakan nyata dengan dampak fisik kuat, bukan sekadar braking di intersection.
                    _det_mode = debug_info.get('detection_mode', 'normal') if debug_info else 'normal'
                    _kelr_i = float(debug_info.get('energy_loss_i', 0.0)) if debug_info else 0.0
                    _kelr_j = float(debug_info.get('energy_loss_j', 0.0)) if debug_info else 0.0
                    _iou_now = float(debug_info.get('iou', 0.0)) if debug_info else 0.0
                    _max_kelr = max(_kelr_i, _kelr_j)
                    # Hard bypass: KELR sangat kuat (impact jelas) DAN overlap besar
                    _bypass_passing_filter = (_max_kelr >= self._PASSING_KELR_BYPASS and
                                              _iou_now >= self._PASSING_IOU_BYPASS)
                    _min_kelr = min(_kelr_i, _kelr_j)
                    _ppl_strong_rescue_hit = (
                        self.ppl_strong_rescue_enabled
                        and _det_mode == 'ppl'
                        and _iou_now >= self.ppl_strong_rescue_iou_min
                        and _min_kelr >= self.ppl_strong_rescue_min_kelr
                    )

                    _ppl_strong_rescue_direct_emitted = False

                    if pair_key not in self.collision_pairs:
                        # First detection — compute passing_candidate
                        if _bypass_passing_filter:
                            _is_passing_cand = False
                        else:
                            _vi = velocity_field.compute_velocity(track_i, dt=1.0)
                            _vj = velocity_field.compute_velocity(track_j, dt=1.0)
                            _si = float(np.linalg.norm(_vi))
                            _sj = float(np.linalg.norm(_vj))
                            if _si > self._PASSING_MIN_SPEED and _sj > self._PASSING_MIN_SPEED:
                                _dot = float(np.dot(_vi / _si, _vj / _sj))
                                _is_passing_cand = _dot < self._PASSING_DOT_THRESH
                            else:
                                _is_passing_cand = False

                        # First detection - enter MONITORING state
                        # Compute baseline speed & AR at moment of contact
                        _vi_now = velocity_field.compute_velocity(track_i, dt=1.0)
                        _vj_now = velocity_field.compute_velocity(track_j, dt=1.0)
                        _speed_i = float(np.linalg.norm(_vi_now))
                        _speed_j = float(np.linalg.norm(_vj_now))

                        def _get_ar(trk):
                            bbox = trk.current_detection.get('bbox')
                            if bbox is None: return None
                            x1,y1,x2,y2 = bbox
                            w = max(float(x2-x1), 1.0); h = max(float(y2-y1), 1.0)
                            return h / w

                        self.collision_pairs[pair_key] = {
                            'state': 'monitoring',
                            'first_detected': current_frame,
                            'persist_count': 1,
                            'last_detected': current_frame,
                            'passing_candidate': _is_passing_cand,
                            'speed_at_contact_i': _speed_i,
                            'speed_at_contact_j': _speed_j,
                            'ar_at_contact_i': _get_ar(track_i),
                            'ar_at_contact_j': _get_ar(track_j),
                            'physical_evidence': False,  # deceleration atau AR change confirmed
                            'ppl_strong_rescue_count': 1 if _ppl_strong_rescue_hit else 0,
                        }
                        # Log first detection
                        detection_mode = debug_info.get('detection_mode', 'normal')
                        _pc_tag = ' [PASSING?]' if _is_passing_cand else ''
                        logger.warning(f"[COLLISION MONITORING] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                     f"State: monitoring (1/{self.persist_window}) | "
                                     f"Mode: {detection_mode.upper()}{_pc_tag} | "
                                     f"IoU: {debug_info.get('iou', 0):.3f} | "
                                     f"Energy Loss: {debug_info.get('energy_loss_i', 0):.3f}, {debug_info.get('energy_loss_j', 0):.3f}")
                    else:
                        # Update persistence count dan last_detected
                        self.collision_pairs[pair_key]['persist_count'] += 1
                        self.collision_pairs[pair_key]['last_detected'] = current_frame

                        # Hard bypass (KELR sangat kuat + IoU besar) → un-mark passing_candidate
                        if _bypass_passing_filter:
                            self.collision_pairs[pair_key]['passing_candidate'] = False
                            self.collision_pairs[pair_key]['physical_evidence'] = True

                        if _ppl_strong_rescue_hit:
                            self.collision_pairs[pair_key]['ppl_strong_rescue_count'] = (
                                self.collision_pairs[pair_key].get('ppl_strong_rescue_count', 0) + 1
                            )
                            if self.collision_pairs[pair_key]['ppl_strong_rescue_count'] >= self.ppl_strong_rescue_frames:
                                self.collision_pairs[pair_key]['passing_candidate'] = False
                                self.collision_pairs[pair_key]['physical_evidence'] = True
                                logger.warning(
                                    f"[PPL STRONG RESCUE] Frame {current_frame}: "
                                    f"Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                    f"count={self.collision_pairs[pair_key]['ppl_strong_rescue_count']} >= {self.ppl_strong_rescue_frames} | "
                                    f"IoU={_iou_now:.3f} >= {self.ppl_strong_rescue_iou_min} | "
                                    f"minKELR={_min_kelr:.3f} >= {self.ppl_strong_rescue_min_kelr} | "
                                    f"Strong repeated PPL evidence accepted as physical evidence"
                                )
                                if not _ppl_strong_rescue_direct_emitted:
                                    bbox_i = track_i.current_detection.get('bbox')
                                    bbox_j = track_j.current_detection.get('bbox')
                                    if bbox_i is not None and bbox_j is not None:
                                        ix1 = max(bbox_i[0], bbox_j[0])
                                        iy1 = max(bbox_i[1], bbox_j[1])
                                        ix2 = min(bbox_i[2], bbox_j[2])
                                        iy2 = min(bbox_i[3], bbox_j[3])
                                        if ix1 < ix2 and iy1 < iy2:
                                            rescue_bbox = [ix1, iy1, ix2, iy2]
                                        else:
                                            rescue_bbox = [
                                                min(bbox_i[0], bbox_j[0]), min(bbox_i[1], bbox_j[1]),
                                                max(bbox_i[2], bbox_j[2]), max(bbox_i[3], bbox_j[3])
                                            ]
                                        rescue_detection = {
                                            'track_id': track_i.track_id,
                                            'track_id_secondary': track_j.track_id,
                                            'frame_id': current_frame,
                                            'detector': self.detector_name,
                                            'behaviour_type': 'collision',
                                            'bbox': rescue_bbox,
                                            'collision_point': self._estimate_collision_point(bbox_i, bbox_j),
                                            'iou_overlap': float(_iou_now),
                                            'energy_loss_primary': float(_kelr_i),
                                            'energy_loss_secondary': float(_kelr_j),
                                            'bbox_primary': bbox_i,
                                            'bbox_secondary': bbox_j,
                                            'confidence': float(self.collision_confidence_threshold),
                                            'confidence_label': 'COLLISION DETECTED',
                                            'alert_level': 'emergency',
                                            'severity': 'critical',
                                            'detection_mode': 'ppl',
                                            'detection_tier': 'ppl_strong_rescue',
                                            'tier': 'ppl_strong_rescue',
                                            'state': 'confirmed',
                                            'persist_count': self.collision_pairs[pair_key].get('persist_count', self.ppl_strong_rescue_frames),
                                            'prediction_mode': 'confirmed',
                                            'persistence': 1.0,
                                            'ppl_primitives': debug_info.get('ppl_primitives', '') if debug_info else '',
                                            'ppl_path': debug_info.get('ppl_path', '') if debug_info else '',
                                            'ars_zscore_i': float(debug_info.get('ars_zscore_i', 0.0)) if debug_info else 0.0,
                                            'ars_zscore_j': float(debug_info.get('ars_zscore_j', 0.0)) if debug_info else 0.0,
                                        }
                                        detections.append(rescue_detection)
                                        _ppl_strong_rescue_direct_emitted = True
                                        logger.warning(
                                            f"[PPL STRONG RESCUE EMIT] Frame {current_frame}: "
                                            f"Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                            f"Level: emergency | Direct final detection emitted from strong repeated PPL evidence"
                                        )

                        # Post-contact physical evidence check (setiap frame)
                        if not self.collision_pairs[pair_key]['physical_evidence']:
                            _spd_i0 = self.collision_pairs[pair_key]['speed_at_contact_i']
                            _spd_j0 = self.collision_pairs[pair_key]['speed_at_contact_j']

                            # Gunakan rata-rata speed 3 frame terakhir dari track history
                            # (bukan instantaneous velocity) untuk resist tracker jitter.
                            # Jitter: speed drop 1 frame lalu balik → avg tetap tinggi → tidak trigger
                            # Tabrakan nyata: speed drop bertahan → avg turun → trigger
                            def _avg_speed(trk, last_n=3):
                                centers = []
                                hist = trk.history
                                for e in hist[-(last_n+1):]:
                                    c = e.get('center')
                                    if c: centers.append(c)
                                if len(centers) < 2: return float(np.linalg.norm(
                                    velocity_field.compute_velocity(trk, dt=1.0)))
                                speeds = []
                                for k in range(1, len(centers)):
                                    dx = centers[k][0] - centers[k-1][0]
                                    dy = centers[k][1] - centers[k-1][1]
                                    speeds.append(float(np.sqrt(dx*dx + dy*dy)))
                                return float(np.mean(speeds)) if speeds else 0.0

                            _si_cur = _avg_speed(track_i)
                            _sj_cur = _avg_speed(track_j)
                            # Evidence 1: deceleration — avg speed turun ke <50% dari saat kontak
                            _decel_i = (_spd_i0 > self._PASSING_MIN_SPEED and
                                        _si_cur < _spd_i0 * self._DECEL_RATIO)
                            _decel_j = (_spd_j0 > self._PASSING_MIN_SPEED and
                                        _sj_cur < _spd_j0 * self._DECEL_RATIO)
                            # Evidence 2: AR deformasi persisten — hitung z-score AR saat ini vs baseline
                            def _ar_zscore(trk, ar_baseline):
                                if ar_baseline is None: return 0.0
                                hist = trk.history
                                if len(hist) < 5: return 0.0
                                ar_vals = []
                                for e in hist[-10:]:
                                    b = e.get('bbox')
                                    if b:
                                        x1,y1,x2,y2 = b
                                        w = max(float(x2-x1),1.0); h = max(float(y2-y1),1.0)
                                        ar_vals.append(h/w)
                                if len(ar_vals) < 3: return 0.0
                                import statistics
                                try:
                                    mu = statistics.mean(ar_vals[:-1])
                                    sd = statistics.stdev(ar_vals[:-1])
                                    if sd < 1e-6: return 0.0
                                    return abs(ar_vals[-1] - mu) / sd
                                except Exception:
                                    return 0.0
                            _arz_i = _ar_zscore(track_i, self.collision_pairs[pair_key]['ar_at_contact_i'])
                            _arz_j = _ar_zscore(track_j, self.collision_pairs[pair_key]['ar_at_contact_j'])
                            _ar_deform = (_arz_i >= self._AR_CHANGE_SIGMA or
                                          _arz_j >= self._AR_CHANGE_SIGMA)
                            if _decel_i or _decel_j or _ar_deform:
                                self.collision_pairs[pair_key]['physical_evidence'] = True
                                logger.warning(f"[PHYSICAL EVIDENCE] Frame {current_frame}: "
                                             f"Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                             f"decel_i={_decel_i}(spd:{_spd_i0:.1f}→{_si_cur:.1f}) "
                                             f"decel_j={_decel_j}(spd:{_spd_j0:.1f}→{_sj_cur:.1f}) "
                                             f"ar_deform={_ar_deform}(z_i={_arz_i:.2f},z_j={_arz_j:.2f})")

                        # Check if should transition to CONFIRMED
                        if (self.collision_pairs[pair_key]['persist_count'] >= self.persist_window and
                            self.collision_pairs[pair_key]['state'] == 'monitoring'):
                            self.collision_pairs[pair_key]['state'] = 'confirmed'
                            _sc_den = round(self.ppl._scene_density, 1)
                            _sc_cat = self.ppl._scene_category
                            logger.warning(f"[COLLISION CONFIRMED] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                         f"State: confirmed ({self.collision_pairs[pair_key]['persist_count']}/{self.persist_window}) | "
                                         f"Scene: {_sc_cat} ({_sc_den} tr/Mpx)")

                    # ── Physical evidence gating ──────────────────────────────────
                    # "Selama belum ada bukti deformasi fisik atau kendaraan berhenti → bukan tabrakan"
                    # Untuk semua detection (passing_candidate atau tidak):
                    #   - Harus ada physical_evidence (decel atau AR change) DALAM _POST_CONTACT_WINDOW frame
                    #   - Atau persist >= _PASSING_CONFIRM_FRAMES (IoU bertahan lama → tabrakan nyata)
                    _pair_data = self.collision_pairs[pair_key]
                    _is_pc = _pair_data.get('passing_candidate', False)
                    _pc_frames = _pair_data['persist_count']
                    _has_evidence = _pair_data.get('physical_evidence', False)
                    _in_window = _pc_frames <= self._POST_CONTACT_WINDOW

                    # "Selama belum ada bukti deformasi fisik atau kendaraan berhenti → bukan tabrakan"
                    # Rule berlaku tanpa pengecualian — tanpa physical_evidence tidak pernah dilaporkan.
                    # Satu-satunya bypass: hard KELR (>0.9) + IoU besar (>0.3) yang set evidence=True langsung.
                    if _ppl_strong_rescue_direct_emitted:
                        pass
                    elif not _has_evidence:
                        if _in_window:
                            logger.warning(f"[EVIDENCE PENDING] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                         f"Waiting for physical evidence ({_pc_frames}/{self._POST_CONTACT_WINDOW} frames)")
                        else:
                            _reason = "passing/crossing vehicle" if _is_pc else "dense traffic / no physical impact"
                            logger.warning(f"[SUPPRESSED] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                         f"No physical evidence in {self._POST_CONTACT_WINDOW} frames → {_reason}")
                    else:
                        # Report if monitoring (for suppress) or confirmed (for visualization)
                        if _pair_data['state'] in ['monitoring', 'confirmed']:
                            # Create detection for this pair (pass debug_info untuk include proximity info)
                            detection = self._create_collision_detection(
                                track_i, track_j, velocity_field, debug_info
                            )

                            if detection is not None:
                                # Add to detections untuk visualization (baik monitoring maupun confirmed)
                                detections.append(detection)

                                # Log berdasarkan state
                                detection_mode = debug_info.get('detection_mode', 'normal')

                                if _pair_data['state'] == 'confirmed':
                                    _alert_level = detection.get('alert_level', 'emergency')
                                    _log_tag = 'COLLISION DETECTED' if _alert_level == 'emergency' else 'COLLISION WARNING'
                                    logger.warning(f"[{_log_tag}] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                                 f"Level: {_alert_level} | "
                                                 f"Mode: {detection_mode.upper()} | "
                                                 f"Path: {debug_info.get('ppl_path', '?')} | "
                                                 f"Strict: {debug_info.get('strict_scene', False)} | "
                                                 f"Primitives: {debug_info.get('ppl_primitives', '?')} | "
                                                 f"IoU: {debug_info.get('iou', 0):.3f} | "
                                                 f"Energy Loss: {debug_info.get('energy_loss_i', 0):.3f}, {debug_info.get('energy_loss_j', 0):.3f} | "
                                                 f"Variance: {debug_info.get('variance_i', 0):.3f}, {debug_info.get('variance_j', 0):.3f}")
                                else:
                                    # Monitoring state - log untuk debug
                                    logger.warning(f"[COLLISION MONITORING] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                                 f"State: monitoring ({_pair_data['persist_count']}/{self.persist_window})")
                else:
                    # No collision detected - check if pair exists and update
                    pair_key = tuple(sorted([track_i.track_id, track_j.track_id]))
                    if pair_key in self.collision_pairs:
                        # Check if should reset (not detected for 5 frames)
                        frames_since = current_frame - self.collision_pairs[pair_key]['last_detected']
                        if frames_since > 5:
                            _was_pc = self.collision_pairs[pair_key].get('passing_candidate', False)
                            _pc_cnt = self.collision_pairs[pair_key]['persist_count']
                            if _was_pc and _pc_cnt < self._PASSING_CONFIRM_FRAMES:
                                logger.warning(f"[PASSING VEHICLE] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                             f"Suppressed (IoU lasted only {_pc_cnt} frames, opposing velocity) — not a collision")
                            else:
                                logger.warning(f"[COLLISION RESET] Frame {current_frame}: Track {track_i.track_id} <-> Track {track_j.track_id} | "
                                             f"Reset (no detection for {frames_since} frames)")
                            del self.collision_pairs[pair_key]

        # ============================================
        # EVASIVE MANEUVER COLLISION DETECTION (NEW!)
        # Check untuk mobil yang tiba-tiba oleng + track yang hilang
        # ============================================
        try:
            logger.debug(f"[EVASIVE] About to call _check_evasive_collision at frame {current_frame}")
            self._check_evasive_collision(tracks, velocity_field, current_frame, detections)
            logger.debug(f"[EVASIVE] Completed _check_evasive_collision at frame {current_frame}")
        except Exception as e:
            logger.error(f"[EVASIVE ERROR] Exception in _check_evasive_collision: {e}", exc_info=True)

        # ============================================
        # POST-IMPACT COLLISION DETECTION (DISABLED)
        # Rollback per user request - causing UI rendering issues
        # ============================================
        # DISABLED

        self.last_collision_detector_time = time.perf_counter() - _profile_detect_start
        self.last_ppl_compute_time = self._profile_ppl_seconds
        self.last_alarm_decision_time = max(
            0.0, self.last_collision_detector_time - self.last_ppl_compute_time
        )
        return detections
    
    def _compute_metric(self, track: object, velocity_field: 'VelocityField') -> float:
        """
        Not used untuk collision (pairwise detection)
        
        Kept for BaseDetector compatibility
        """
        return 0.0
    
    def _check_condition(self, metric_value: float, track: object) -> bool:
        """
        Not used untuk collision (pairwise detection)
        
        Kept for BaseDetector compatibility
        """
        return False
    
    def _check_pairwise_collision(self, track_i: object, track_j: object, 
                                  velocity_field: 'VelocityField',
                                  adaptive_threshold: float = None) -> Tuple[bool, Dict]:
        """
        Check collision between two tracks dengan INSTANT detection saat overlap
        
        Args:
            track_i: First track
            track_j: Second track
            velocity_field: VelocityField
            adaptive_threshold: Adaptive energy loss threshold (default: use self.energy_loss_threshold)
            
        Returns:
            (bool, dict): (True jika collision detected, debug_info)
        """
        if adaptive_threshold is None:
            adaptive_threshold = self.energy_loss_threshold

        # ============================================
        # CLASS-BASED FILTER: vehicle↔vehicle pairs only
        # Skip any pair that involves person/pedestrian — termasuk person+motorcycle.
        # "Mobil vs motor" = car↔motorcycle (objek kendaraan), bukan person↔vehicle.
        # Menggunakan _is_vehicle_vehicle_pair yang sudah benar baca current_detection.
        # ============================================
        if not self._is_vehicle_vehicle_pair(track_i, track_j):
            class_i_dbg = track_i.current_detection.get('class_name', 'unknown') if hasattr(track_i, 'current_detection') else 'unknown'
            class_j_dbg = track_j.current_detection.get('class_name', 'unknown') if hasattr(track_j, 'current_detection') else 'unknown'
            logger.debug(f"  ⏭️ SKIP collision check | Track {track_i.track_id} ({class_i_dbg}) + Track {track_j.track_id} ({class_j_dbg}) | "
                        f"Not vehicle+vehicle pair — only car/truck/bus/motorcycle/bicycle pairs are checked")
            return False, {'iou': 0.0, 'energy_loss_i': 0.0, 'energy_loss_j': 0.0,
                          'variance_i': 0.0, 'variance_j': 0.0,
                          'detection_mode': 'skipped_non_vehicle'}

        # ============================================
        # MINIMUM TRACK QUALITY FILTER
        # Ghost/short-lived tracks (hits < min_track_hits) cause most FPs in Smart Hungarian
        # Real vehicles have hits >> 5, ghost tracks typically hits 1-3
        # ============================================
        hits_i = getattr(track_i, 'hits', 999)
        hits_j = getattr(track_j, 'hits', 999)
        if hits_i < self.min_track_hits or hits_j < self.min_track_hits:
            logger.debug(f"  ⏭️ SKIP collision check | Track {track_i.track_id} (H:{hits_i}) + Track {track_j.track_id} (H:{hits_j}) | "
                        f"Below min_track_hits={self.min_track_hits}")
            return False, {'iou': 0.0, 'energy_loss_i': 0.0, 'energy_loss_j': 0.0,
                          'variance_i': 0.0, 'variance_j': 0.0,
                          'detection_mode': 'skipped_low_hits'}

        # Compute is_vehicle_vehicle (used later for tier filtering)
        is_vehicle_vehicle = self._is_vehicle_vehicle_pair(track_i, track_j)

        # ============================================
        # GHOST TRACK GUARD (FALSE POSITIVE PREVENTION)
        # Jika salah satu track adalah ghost (misses > 0), bbox-nya stale dari frame sebelumnya.
        # Sumber FP utama:
        #   - PPL PERSIST: IoU dihitung dari bbox stale, seolah overlap fisik padahal kendaraan sudah tidak ada
        #   - PPL ARS: shape change dari track aktif lain + stale bbox ghost = false trigger
        #   - Tier heuristics: semua tier menggunakan IoU yang tidak valid dari bbox stale
        #
        # Guard dipasang SEBELUM PPL agar PPL tidak mengonsumsi bbox stale sama sekali.
        # Genuine post-collision: PPL sudah selesai di frame di mana kedua track masih aktif.
        # Setelah salah satu menjadi ghost, PPL tidak akan membantu dan justru menambah FP.
        # ============================================
        _track_i_misses = getattr(track_i, 'misses', 0)
        _track_j_misses = getattr(track_j, 'misses', 0)
        if _track_i_misses > 0 or _track_j_misses > 0:
            logger.debug(
                f"  ⏭️ SKIP (GHOST TRACK) | "
                f"Track {track_i.track_id} (misses={_track_i_misses}) <-> "
                f"Track {track_j.track_id} (misses={_track_j_misses}) | "
                f"Bbox stale — skipping PPL + all tiers to prevent false positive"
            )
            return False, {
                'iou': 0.0, 'energy_loss_i': 0.0, 'energy_loss_j': 0.0,
                'variance_i': 0.0, 'variance_j': 0.0,
                'detection_mode': 'skipped_ghost_track'
            }

        # ============================================
        # PHYSICS PRIMITIVE LAYER (PPL) — Priority Override
        # Tracker-agnostic oracle: CE + KELR_rel + ARS (dimensionless, scale-free).
        # If ≥2 primitives fire → collision is physics-confirmed; skip all tiers.
        # Class and quality filters above already passed, so PPL only runs on
        # legitimate track pairs.
        # ============================================
        _ppl_frame_id = getattr(track_i, 'current_frame', 0)
        _ppl_profile_start = time.perf_counter()
        try:
            _ppl_result = self.ppl.check_pair(track_i, track_j, _ppl_frame_id)
        finally:
            self._profile_ppl_seconds += time.perf_counter() - _ppl_profile_start
        if _ppl_result is not None:
            _fired = "+".join(_ppl_result['primitives_fired'])
            logger.warning(
                f"[PPL COLLISION] Frame {_ppl_frame_id}: "
                f"Track {track_i.track_id} <-> Track {track_j.track_id} | "
                f"Primitives fired: {_fired} | "
                f"CE={_ppl_result['primitive_ce']} "
                f"KELR={_ppl_result['primitive_kelr']} "
                f"(val_i={_ppl_result['kelr_value_i']:.3f}, val_j={_ppl_result['kelr_value_j']:.3f}) "
                f"ARS={_ppl_result['primitive_ars']} "
                f"(z_i={_ppl_result['ars_zscore_i']:.2f}, z_j={_ppl_result['ars_zscore_j']:.2f}) "
                f"IoU={_ppl_result['iou_now']:.3f} | "
                f"Physics-confirmed collision (bypassing multi-tier heuristics)"
            )
            _iou_now = self._compute_iou(
                track_i.current_detection['bbox'],
                track_j.current_detection['bbox']
            )
            return True, {
                'iou': _iou_now,
                'energy_loss_i': _ppl_result['kelr_value_i'],
                'energy_loss_j': _ppl_result['kelr_value_j'],
                'ars_zscore_i':  _ppl_result['ars_zscore_i'],
                'ars_zscore_j':  _ppl_result['ars_zscore_j'],
                'variance_i': 0.0,
                'variance_j': 0.0,
                'detection_mode': 'ppl',
                'detection_tier': 'ppl',
                'tier': 'ppl',
                'ppl_primitives': _fired,
                'ppl_path': _ppl_result.get('path_triggered', 'unknown'),
                'ppl_kelr_value_i': _ppl_result['kelr_value_i'],
                'ppl_kelr_value_j': _ppl_result['kelr_value_j'],
                'ppl_iou_now': _ppl_result['iou_now'],
                'strict_scene': _ppl_result.get('strict_scene', False),
            }

        # PPL did not fire — no collision
        return False, {'iou': 0.0, 'energy_loss_i': 0.0, 'energy_loss_j': 0.0,
                        'variance_i': 0.0, 'variance_j': 0.0, 'detection_mode': 'no_detection'}

    def _compute_collision_point(self, bbox_i: List, bbox_j: List) -> List[int]:
        """
        Compute collision point (center point between two bboxes)

        Args:
            bbox_i: Bounding box [x1, y1, x2, y2]
            bbox_j: Bounding box [x1, y1, x2, y2]

        Returns:
            Collision point [x, y]
        """
        # Calculate center points
        center_i = [(bbox_i[0] + bbox_i[2]) / 2, (bbox_i[1] + bbox_i[3]) / 2]
        center_j = [(bbox_j[0] + bbox_j[2]) / 2, (bbox_j[1] + bbox_j[3]) / 2]

        # Collision point is midpoint between centers
        collision_point = [
            int((center_i[0] + center_j[0]) / 2),
            int((center_i[1] + center_j[1]) / 2)
        ]

        return collision_point

    def _is_vehicle(self, track: object) -> bool:
        """
        Check apakah track adalah vehicle (car, truck, bus, etc.)
        Note: motorcycle dan bicycle NOT included untuk evasive detection

        Args:
            track: Track object

        Returns:
            True jika vehicle (car/truck/bus), False jika bukan
        """
        class_name = track.current_detection.get('class_name', 'unknown')
        # Only car, truck, bus, van - NOT motorcycle/bicycle for evasive maneuver
        vehicle_classes = ['car', 'truck', 'bus', 'van', 'vehicle']
        return class_name.lower() in vehicle_classes

    def _is_vehicle_class(self, class_name: str) -> bool:
        """
        Check apakah class adalah vehicle (car, truck, bus, motorcycle, etc.)

        Args:
            class_name: Class name dari detection

        Returns:
            True jika vehicle, False jika bukan
        """
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'van', 'vehicle']
        return class_name.lower() in vehicle_classes
    
    def _is_pedestrian_class(self, class_name: str) -> bool:
        """
        Check apakah class adalah pedestrian (person, pedestrian, etc.)
        
        Args:
            class_name: Class name dari detection
            
        Returns:
            True jika pedestrian, False jika bukan
        """
        pedestrian_classes = ['person', 'pedestrian', 'people']
        return class_name.lower() in pedestrian_classes
    
    def _is_person_motorcycle_pair(self, class_a: str, class_b: str) -> bool:
        """
        Check apakah pair adalah person + motorcycle (rider on motorcycle).
        Pair ini TIDAK di-skip karena di dataset collision sering melibatkan motor + rider.
        """
        a = class_a.lower()
        b = class_b.lower()
        return (self._is_pedestrian_class(a) and 'motorcycle' in b) or \
               (self._is_pedestrian_class(b) and 'motorcycle' in a)

    def _is_vehicle_vehicle_pair(self, track_i: object, track_j: object) -> bool:
        """
        Check apakah pair adalah vehicle-vehicle
        
        Args:
            track_i: First track
            track_j: Second track
            
        Returns:
            True jika kedua track adalah vehicle
        """
        class_i = track_i.current_detection.get('class_name', 'unknown')
        class_j = track_j.current_detection.get('class_name', 'unknown')
        return self._is_vehicle_class(class_i) and self._is_vehicle_class(class_j)
    
    def _is_vehicle_pedestrian_pair(self, track_i: object, track_j: object) -> bool:
        """
        Check apakah pair adalah vehicle-pedestrian
        
        Args:
            track_i: First track
            track_j: Second track
            
        Returns:
            True jika salah satu vehicle dan satu pedestrian
        """
        class_i = track_i.current_detection.get('class_name', 'unknown')
        class_j = track_j.current_detection.get('class_name', 'unknown')
        return (self._is_vehicle_class(class_i) and self._is_pedestrian_class(class_j)) or \
               (self._is_pedestrian_class(class_i) and self._is_vehicle_class(class_j))
    
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
    
    def _compute_energy_loss(self, track: object, velocity_field: 'VelocityField', 
                            frames_back: int = 3) -> float:
        """
        Compute kinetic energy loss ratio (Blueprint Section 2.4.5)
        
        Energy Loss = (E_past - E_current) / E_past
        
        Args:
            track: Track object
            velocity_field: VelocityField
            frames_back: How many frames to look back
            
        Returns:
            Energy loss ratio [0, 1]
        """
        if len(track.history) <= frames_back:
            return 0.0

        # Current kinetic energy
        v_current = velocity_field.compute_velocity(track, dt=1.0)
        speed_current = np.linalg.norm(v_current)
        bbox_current = track.current_detection['bbox']
        mass_current = (bbox_current[2] - bbox_current[0]) * (bbox_current[3] - bbox_current[1])
        E_current = 0.5 * mass_current * speed_current**2

        # Past kinetic energy (frames_back ago)
        original_current = track.current_detection
        track.current_detection = track.history[-(frames_back + 1)]
        v_past = velocity_field.compute_velocity(track, dt=1.0)
        track.current_detection = original_current

        speed_past = np.linalg.norm(v_past)
        bbox_past = track.history[-(frames_back + 1)]['bbox']
        mass_past = (bbox_past[2] - bbox_past[0]) * (bbox_past[3] - bbox_past[1])
        E_past = 0.5 * mass_past * speed_past**2

        # CRITICAL FIX: Static velocity check (v ≈ 0)
        # Jika kedua velocity (current & past) ≈ 0 → energy loss = 0
        STATIC_VELOCITY_THRESHOLD = 0.5  # px/fr (increased from 0.3 for consistency)
        if speed_current < STATIC_VELOCITY_THRESHOLD and speed_past < STATIC_VELOCITY_THRESHOLD:
            return 0.0

        # Energy loss ratio
        if E_past < 1e-6:
            return 0.0
        
        energy_loss = (E_past - E_current) / E_past
        
        return max(0.0, energy_loss)  # Clip to [0, inf]
    
    def _compute_energy_loss_instant(self, track: object, velocity_field: 'VelocityField',
                                    threshold: float) -> float:
        """
        Compute energy loss dengan INSTANT detection untuk track baru (H:1)

        ENHANCED: Untuk track baru, gunakan multiple indicators:
        1. Velocity drop instan (v_current < 0.1 * v_expected)
        2. Acceleration spike (sudden deceleration = collision indicator)
        3. Rotation spike (sudden spin = collision indicator)

        Args:
            track: Track object
            velocity_field: VelocityField
            threshold: Energy loss threshold untuk determine if instant drop is significant

        Returns:
            Energy loss ratio [0, 1] atau 1.0 jika instant drop detected
        """
        # Try normal energy loss calculation first
        if len(track.history) > self.energy_loss_frames_back:
            return self._compute_energy_loss(track, velocity_field, frames_back=self.energy_loss_frames_back)

        # Track baru (H:1) - gunakan multiple indicators
        v_current = velocity_field.compute_velocity(track, dt=1.0)
        speed_current = np.linalg.norm(v_current)

        # CRITICAL FIX: Static velocity check (v ≈ 0)
        # Jika velocity ≈ 0, energy loss = 0 (KE = 0.5 * m * v^2 → v=0 berarti KE=0)
        STATIC_VELOCITY_THRESHOLD = 0.5  # px/fr (increased from 0.3 for consistency)
        if speed_current < STATIC_VELOCITY_THRESHOLD:
            # Check velocity history juga
            if len(track.history) >= 1:
                original = track.current_detection
                track.current_detection = track.history[-1]
                v_past = velocity_field.compute_velocity(track, dt=1.0)
                track.current_detection = original
                speed_past = np.linalg.norm(v_past)

                # Jika current dan past velocity KEDUA-nya rendah → no energy loss
                if speed_past < STATIC_VELOCITY_THRESHOLD:
                    return 0.0
            else:
                # No history - jika current v ≈ 0, return 0
                return 0.0
        
        # ENHANCED: Check acceleration spike sebagai alternatif indicator
        # Compute acceleration spike untuk single track
        acceleration_spike = 0.0
        if len(track.history) >= self.acceleration_spike_frames_back:
            original = track.current_detection
            track.current_detection = track.history[-self.acceleration_spike_frames_back]
            v_past = velocity_field.compute_velocity(track, dt=1.0)
            track.current_detection = original
            
            # Acceleration: a = Δv / Δt
            delta_v = v_current - v_past
            acceleration_spike = np.linalg.norm(delta_v) / self.acceleration_spike_frames_back
        
        rotation_spike = self._compute_rotation_spike(track, velocity_field)
        
        # Jika ada acceleration spike tinggi atau rotation spike → likely collision
        if acceleration_spike > self.acceleration_spike_threshold or rotation_spike > 0.5:
            # Strong collision indicator untuk track baru
            return 0.8  # High energy loss (80%) untuk track baru dengan spike
        
        # Get expected velocity dari scene (velocity_field global)
        # Build field dari all tracks untuk get average scene velocity
        if hasattr(velocity_field, 'cached_field') and velocity_field.cached_field is not None:
            field_data = velocity_field.cached_field
            if len(field_data.get('velocities', [])) > 0:
                # Compute average scene velocity
                velocities = field_data['velocities']
                avg_velocity = np.mean(velocities, axis=0)
                v_expected = np.linalg.norm(avg_velocity)
                
                # Instant drop: v_current < 0.1 * v_expected
                if v_expected > 1e-6 and speed_current < 0.1 * v_expected:
                    # Significant instant drop detected
                    return 1.0  # Return maximum energy loss
                
                # Or use ratio if available
                if v_expected > 1e-6:
                    energy_loss_ratio = 1.0 - (speed_current / v_expected)
                    return max(0.0, min(1.0, energy_loss_ratio))
        
        # Fallback: jika tidak ada scene context, return 0 (no energy loss detected)
        return 0.0
    
    def _get_adaptive_threshold(self) -> float:
        """
        Get adaptive energy loss threshold berdasarkan scene density
        
        Blueprint strict: >0.9 untuk normal scene
        Adaptive: >0.7 untuk sparse scene
        
        Returns:
            Adaptive threshold
        """
        if self.scene_analyzer is None:
            return self.energy_loss_threshold  # Default: 0.9
        
        # Get scene category dari scene analyzer
        scene_info = getattr(self.scene_analyzer, 'current_category', 'normal')
        density = getattr(self.scene_analyzer, 'current_density', 10.0)
        
        # Adaptive threshold: sparse scene (< 5 tr/Mpx) -> 0.7, normal -> 0.9
        if scene_info == 'sparse' or density < 5.0:
            return 0.7  # Relaxed untuk sparse scene
        else:
            return 0.9  # Blueprint strict untuk normal/dense scene
    
    def _get_expected_velocity(self, velocity_field: 'VelocityField') -> float:
        """
        Get expected velocity dari scene untuk velocity drop detection
        
        Args:
            velocity_field: VelocityField
            
        Returns:
            Expected velocity magnitude (px/frame)
        """
        if hasattr(velocity_field, 'cached_field') and velocity_field.cached_field is not None:
            field_data = velocity_field.cached_field
            if len(field_data.get('velocities', [])) > 0:
                velocities = field_data['velocities']
                avg_velocity = np.mean(velocities, axis=0)
                v_expected = np.linalg.norm(avg_velocity)
                return v_expected
        
        # Fallback: default expected velocity (assume 10 px/frame untuk normal traffic)
        return 10.0
    
    def _update_track_baselines(self, tracks: List):
        """
        Update baseline AR dan area untuk setiap track
        
        Baseline digunakan untuk detect deformasi (perubahan bentuk).
        Baseline = average AR dan area dari history frames.
        
        Args:
            tracks: List of track objects
        """
        for track in tracks:
            track_id = track.track_id
            
            # Skip jika track tidak valid
            if not self._is_valid_track(track):
                continue
            
            # Initialize baseline jika belum ada
            if track_id not in self.track_baselines:
                self.track_baselines[track_id] = {
                    'ar_baseline': None,
                    'area_baseline': None,
                    'baseline_frames': []
                }
            
            # Collect AR dan area dari history
            baseline_data = self.track_baselines[track_id]
            baseline_frames = baseline_data['baseline_frames']
            
            # Get current AR dan area
            current_bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
            current_ar = self._compute_aspect_ratio(current_bbox)
            current_area = self._compute_area(current_bbox)
            
            # Add to baseline frames (keep last N frames)
            baseline_frames.append({
                'ar': current_ar,
                'area': current_area
            })
            
            # Keep only last baseline_frames frames
            if len(baseline_frames) > self.baseline_frames:
                baseline_frames.pop(0)
            
            # Update baseline (average dari collected frames)
            if len(baseline_frames) >= 3:  # Minimum 3 frames untuk baseline
                ar_values = [f['ar'] for f in baseline_frames]
                area_values = [f['area'] for f in baseline_frames]
                
                baseline_data['ar_baseline'] = np.mean(ar_values)
                baseline_data['area_baseline'] = np.mean(area_values)
    
    def _compute_aspect_ratio(self, bbox: List[float]) -> float:
        """
        Compute aspect ratio dari bbox

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            Aspect ratio (height / width)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if width < 1e-6:
            return 0.0

        return height / width

    def _check_ar_inversion(self, track: object) -> Tuple[bool, float, float]:
        """
        Check aspect ratio inversion untuk motorcycle (motor jatuh)

        AR Inversion: Motor berdiri (AR > 1.2) → Motor jatuh (AR < 0.8)

        Logic diambil dari fallen_detector.py untuk detect motor jatuh setelah collision

        Args:
            track: Track object

        Returns:
            (is_inversion, ar_past, ar_current)
            - is_inversion: True jika AR inversion terdeteksi
            - ar_past: Aspect ratio sebelumnya
            - ar_current: Aspect ratio sekarang
        """
        # FILTER: Hanya untuk motorcycle
        class_name = track.current_detection.get('class_name', 'unknown')
        if class_name != 'motorcycle':
            return False, 0.0, 0.0

        # Need minimal 3 frames history untuk detect AR flip
        if not hasattr(track, 'history') or len(track.history) < 3:
            return False, 0.0, 0.0

        # Current aspect ratio
        current_bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
        ar_current = self._compute_aspect_ratio(current_bbox)

        # Past aspect ratio (3 frames ago)
        past_bbox = track.history[-3].get('bbox', [0, 0, 0, 0])
        ar_past = self._compute_aspect_ratio(past_bbox)

        # Check AR inversion: standing (AR > 1.2) → fallen (AR < 0.8)
        # Threshold dari fallen_detector.py
        is_inversion = (ar_past > 1.2 and ar_current < 0.8)

        return is_inversion, ar_past, ar_current
    
    def _compute_area(self, bbox: List[float]) -> float:
        """
        Compute area dari bbox
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Area (width * height)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        return width * height
    
    def _compute_deformation(self, track: object) -> Tuple[bool, float, float]:
        """
        Compute deformation (perubahan bentuk) untuk track
        
        Deformation = perubahan AR atau area yang signifikan dari baseline.
        Mobil yang tabrakan akan mengalami deformasi (ringsek).
        Mobil yang hanya berdekatan: bentuk tetap sama → tidak ada deformasi.
        
        Args:
            track: Track object
            
        Returns:
            (deformation_detected, ar_change_ratio, area_change_ratio)
            - deformation_detected: True jika AR change > threshold ATAU area change > threshold
            - ar_change_ratio: Percentage change in AR (0.0 = no change, 0.2 = 20% change)
            - area_change_ratio: Percentage change in area
        """
        track_id = track.track_id
        
        # Check if baseline exists
        if track_id not in self.track_baselines:
            return False, 0.0, 0.0
        
        baseline_data = self.track_baselines[track_id]
        ar_baseline = baseline_data.get('ar_baseline')
        area_baseline = baseline_data.get('area_baseline')
        
        # ENHANCED: Untuk track baru (baseline belum ready), gunakan predicted baseline
        # atau gunakan history frames yang tersedia untuk estimate baseline
        if ar_baseline is None or area_baseline is None:
            # Track baru - coba gunakan history yang tersedia untuk estimate baseline
            baseline_frames = baseline_data.get('baseline_frames', [])
            if len(baseline_frames) >= 1:
                # Gunakan average dari frames yang tersedia sebagai baseline
                ar_values = [f.get('ar', 0) for f in baseline_frames if 'ar' in f]
                area_values = [f.get('area', 0) for f in baseline_frames if 'area' in f]
                
                if len(ar_values) > 0 and len(area_values) > 0:
                    ar_baseline = np.mean(ar_values)
                    area_baseline = np.mean(area_values)
                else:
                    return False, 0.0, 0.0
            else:
                # Tidak ada history sama sekali - tidak bisa detect deformation
                return False, 0.0, 0.0
        
        # Get current AR dan area
        current_bbox = track.current_detection.get('bbox', [0, 0, 0, 0])
        current_ar = self._compute_aspect_ratio(current_bbox)
        current_area = self._compute_area(current_bbox)
        
        # Compute change ratios
        if ar_baseline > 1e-6:
            ar_change_ratio = abs(current_ar - ar_baseline) / ar_baseline
        else:
            ar_change_ratio = 0.0
        
        if area_baseline > 1e-6:
            area_change_ratio = abs(current_area - area_baseline) / area_baseline
        else:
            area_change_ratio = 0.0
        
        # Deformation detected jika AR change > threshold ATAU area change > threshold
        deformation_detected = (
            ar_change_ratio > self.ar_change_threshold or
            area_change_ratio > self.area_change_threshold
        )
        
        return deformation_detected, ar_change_ratio, area_change_ratio
    
    def _compute_rotation_spike(self, track: object, velocity_field: 'VelocityField') -> float:
        """
        Compute rotation spike (sudden spin) untuk collision detection
        
        Prinsip: Mobil belok normal = vorticity berubah gradual (smooth)
                 Mobil tabrakan = vorticity spike (sudden, chaotic spin)
        
        Rotation Spike = |Vorticity(t) - Vorticity(t-2)| > threshold
        
        Args:
            track: Track object
            velocity_field: VelocityField untuk compute vorticity (cross product)
            
        Returns:
            Rotation spike magnitude (0.0 = no spike, > 0.8 = sudden spin = collision)
        """
        # Need at least 3 frames untuk compute vorticity change
        if not hasattr(track, 'history') or len(track.history) < 3:
            return 0.0
        
        # Compute current vorticity (cross product 2D)
        vorticity_current = velocity_field.compute_cross_product_2d(track)
        
        # Compute vorticity 2 frames ago (untuk detect sudden change)
        if len(track.history) >= 3:
            # Save current detection
            original_current = track.current_detection
            
            # Get detection 2 frames ago
            track.current_detection = track.history[-3]
            vorticity_past = velocity_field.compute_cross_product_2d(track)
            
            # Restore current detection
            track.current_detection = original_current
            
            # Rotation spike = sudden change in vorticity
            # Turn normal: |ΔVorticity| ≈ 0.1-0.3 per frame (gradual)
            # Collision spin: |ΔVorticity| > 0.8 per frame (sudden)
            rotation_spike = abs(vorticity_current - vorticity_past)
            
            return rotation_spike
        
        return 0.0
    
    def _compute_momentum_transfer(self, track_i: object, track_j: object,
                                  velocity_field: 'VelocityField') -> Tuple[float, float]:
        """
        Compute momentum transfer untuk push collision detection (rear-end)
        
        Prinsip: Ketika mobil belakang (A) mendorong mobil depan (B):
        - Mobil B (depan) tiba-tiba accelerate: mendapat momentum dari A
        - Mobil A (belakang) tiba-tiba decelerate: kehilangan momentum
        
        Momentum Transfer = m · |Δv|
        
        ENHANCED: Validasi konsistensi fisika untuk mencegah false positive
        - Track baru (history < 5 frames) → tidak reliable, return 0
        - Velocity ≈ 0 tapi momentum tinggi → tidak konsisten, cap atau reject
        - Normalize berdasarkan velocity magnitude untuk mencegah bbox size bias
        
        Args:
            track_i: First track (biasanya mobil belakang)
            track_j: Second track (biasanya mobil depan)
            velocity_field: VelocityField
            
        Returns:
            (momentum_transfer_i, momentum_transfer_j): Momentum transfer magnitude
        """
        momentum_i = 0.0
        momentum_j = 0.0

        # Minimum history untuk reliable momentum transfer calculation
        min_history_frames = 5

        # CRITICAL FIX: Static velocity check (v ≈ 0)
        # Jika kedua velocity ≈ 0, momentum transfer = 0 (fisika dasar)
        v_current_i = velocity_field.compute_velocity(track_i, dt=1.0)
        v_current_j = velocity_field.compute_velocity(track_j, dt=1.0)
        v_current_speed_i = np.linalg.norm(v_current_i)
        v_current_speed_j = np.linalg.norm(v_current_j)

        STATIC_VELOCITY_THRESHOLD = 0.5  # px/fr (increased from 0.3 for consistency)
        if v_current_speed_i < STATIC_VELOCITY_THRESHOLD and v_current_speed_j < STATIC_VELOCITY_THRESHOLD:
            # Kedua objek stationary - no momentum transfer possible
            return 0.0, 0.0

        # Compute momentum transfer untuk track_i
        if len(track_i.history) >= max(self.push_frames_back, min_history_frames):
            # Re-use v_current_i from above (already computed)
            
            original = track_i.current_detection
            track_i.current_detection = track_i.history[-self.push_frames_back]
            v_past_i = velocity_field.compute_velocity(track_i, dt=1.0)
            track_i.current_detection = original
            
            v_past_speed_i = np.linalg.norm(v_past_i)
            
            # Mass approximation dari bbox area
            bbox_i = track_i.current_detection['bbox']
            mass_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
            
            # Momentum transfer: m · |Δv|
            delta_v_i = v_current_i - v_past_i
            delta_v_magnitude_i = np.linalg.norm(delta_v_i)
            momentum_i = mass_i * delta_v_magnitude_i
            
            # VALIDASI KONSISTENSI FISIKA:
            # 1. Jika velocity current ≈ 0, momentum transfer harus kecil
            #    (tidak mungkin ada momentum transfer besar jika objek stationary)
            if v_current_speed_i < 0.5:  # Velocity sangat rendah (< 0.5 px/fr)
                # Cap momentum transfer berdasarkan velocity magnitude
                # Momentum max = mass * v_max, dimana v_max = max(v_current, v_past)
                max_velocity_i = max(v_current_speed_i, v_past_speed_i)
                max_valid_momentum_i = mass_i * max_velocity_i * 2.0  # Allow 2x untuk safety margin
                momentum_i = min(momentum_i, max_valid_momentum_i)
            
            # 2. Jika velocity change sangat kecil, momentum transfer harus kecil
            #    (mencegah false positive dari noise)
            if delta_v_magnitude_i < 0.1:  # Velocity change sangat kecil
                momentum_i = 0.0  # Reject sebagai noise
            
            # 3. Normalize: momentum transfer tidak boleh terlalu besar relatif terhadap velocity
            #    Jika momentum > mass * v_max * 3, kemungkinan error
            max_velocity_i = max(v_current_speed_i, v_past_speed_i)
            if max_velocity_i > 0.1:  # Hanya jika ada velocity yang signifikan
                max_reasonable_momentum_i = mass_i * max_velocity_i * 3.0
                if momentum_i > max_reasonable_momentum_i:
                    # Cap to reasonable value
                    momentum_i = max_reasonable_momentum_i
        
        # Compute momentum transfer untuk track_j
        if len(track_j.history) >= max(self.push_frames_back, min_history_frames):
            v_current_j = velocity_field.compute_velocity(track_j, dt=1.0)
            v_current_speed_j = np.linalg.norm(v_current_j)
            
            original = track_j.current_detection
            track_j.current_detection = track_j.history[-self.push_frames_back]
            v_past_j = velocity_field.compute_velocity(track_j, dt=1.0)
            track_j.current_detection = original
            
            v_past_speed_j = np.linalg.norm(v_past_j)
            
            # Mass approximation
            bbox_j = track_j.current_detection['bbox']
            mass_j = (bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1])
            
            # Momentum transfer: m · |Δv|
            delta_v_j = v_current_j - v_past_j
            delta_v_magnitude_j = np.linalg.norm(delta_v_j)
            momentum_j = mass_j * delta_v_magnitude_j
            
            # VALIDASI KONSISTENSI FISIKA (sama seperti track_i):
            # 1. Jika velocity current ≈ 0, momentum transfer harus kecil
            if v_current_speed_j < 0.5:
                max_velocity_j = max(v_current_speed_j, v_past_speed_j)
                max_valid_momentum_j = mass_j * max_velocity_j * 2.0
                momentum_j = min(momentum_j, max_valid_momentum_j)
            
            # 2. Jika velocity change sangat kecil, reject sebagai noise
            if delta_v_magnitude_j < 0.1:
                momentum_j = 0.0
            
            # 3. Normalize: cap jika terlalu besar relatif terhadap velocity
            max_velocity_j = max(v_current_speed_j, v_past_speed_j)
            if max_velocity_j > 0.1:
                max_reasonable_momentum_j = mass_j * max_velocity_j * 3.0
                if momentum_j > max_reasonable_momentum_j:
                    momentum_j = max_reasonable_momentum_j
        
        return momentum_i, momentum_j
    
    def _compute_relative_velocity_drop(self, track_i: object, track_j: object,
                                       velocity_field: 'VelocityField') -> float:
        """
        Compute relative velocity drop untuk push collision detection
        
        Prinsip: Sebelum collision, relative velocity besar (A lebih cepat dari B)
                 Setelah collision, relative velocity drop drastis (keduanya bergerak bersama)
        
        Relative Velocity Drop = (v_rel_before - v_rel_after) / v_rel_before
        
        Args:
            track_i: First track
            track_j: Second track
            velocity_field: VelocityField
            
        Returns:
            Relative velocity drop ratio [0, 1]
        """
        if len(track_i.history) < self.push_frames_back or len(track_j.history) < self.push_frames_back:
            return 0.0
        
        # Current relative velocity
        v_current_i = velocity_field.compute_velocity(track_i, dt=1.0)
        v_current_j = velocity_field.compute_velocity(track_j, dt=1.0)
        v_rel_current = np.linalg.norm(v_current_i - v_current_j)
        
        # Past relative velocity (push_frames_back ago)
        original_i = track_i.current_detection
        original_j = track_j.current_detection
        
        track_i.current_detection = track_i.history[-self.push_frames_back]
        track_j.current_detection = track_j.history[-self.push_frames_back]
        
        v_past_i = velocity_field.compute_velocity(track_i, dt=1.0)
        v_past_j = velocity_field.compute_velocity(track_j, dt=1.0)
        v_rel_past = np.linalg.norm(v_past_i - v_past_j)
        
        # Restore
        track_i.current_detection = original_i
        track_j.current_detection = original_j
        
        # Relative velocity drop
        if v_rel_past < 1e-6:
            return 0.0
        
        relative_velocity_drop = (v_rel_past - v_rel_current) / v_rel_past
        return max(0.0, min(1.0, relative_velocity_drop))  # Clip to [0, 1]
    
    def _compute_push_acceleration(self, track_i: object, track_j: object,
                                  velocity_field: 'VelocityField') -> Tuple[float, float]:
        """
        Compute push acceleration untuk push collision detection
        
        Prinsip: Mobil depan (B) tiba-tiba accelerate karena didorong
                 Mobil belakang (A) tiba-tiba decelerate karena mendorong
        
        Push Acceleration = |Δv| / Δt
        
        Args:
            track_i: First track
            track_j: Second track
            velocity_field: VelocityField
            
        Returns:
            (push_acceleration_i, push_acceleration_j): Acceleration magnitude
        """
        acc_i = 0.0
        acc_j = 0.0
        
        # Compute push acceleration untuk track_i
        if len(track_i.history) >= self.push_frames_back:
            v_current_i = velocity_field.compute_velocity(track_i, dt=1.0)
            original = track_i.current_detection
            track_i.current_detection = track_i.history[-self.push_frames_back]
            v_past_i = velocity_field.compute_velocity(track_i, dt=1.0)
            track_i.current_detection = original
            
            # Push acceleration: |Δv| / Δt
            delta_v_i = v_current_i - v_past_i
            acc_i = np.linalg.norm(delta_v_i) / self.push_frames_back
        
        # Compute push acceleration untuk track_j
        if len(track_j.history) >= self.push_frames_back:
            v_current_j = velocity_field.compute_velocity(track_j, dt=1.0)
            original = track_j.current_detection
            track_j.current_detection = track_j.history[-self.push_frames_back]
            v_past_j = velocity_field.compute_velocity(track_j, dt=1.0)
            track_j.current_detection = original
            
            # Push acceleration: |Δv| / Δt
            delta_v_j = v_current_j - v_past_j
            acc_j = np.linalg.norm(delta_v_j) / self.push_frames_back
        
        return acc_i, acc_j
    
    def _compute_velocity_direction_change(self, track_i: object, track_j: object,
                                         velocity_field: 'VelocityField') -> Tuple[float, float]:
        """
        Compute velocity direction change untuk push collision detection
        
        Prinsip: Mobil depan (B) tiba-tiba berubah arah karena didorong
        
        Direction Change = angle_between(v_current, v_past)
        
        Args:
            track_i: First track
            track_j: Second track
            velocity_field: VelocityField
            
        Returns:
            (direction_change_i, direction_change_j): Direction change in degrees
        """
        direction_change_i = 0.0
        direction_change_j = 0.0
        
        # Compute direction change untuk track_i
        if len(track_i.history) >= self.push_frames_back:
            v_current_i = velocity_field.compute_velocity(track_i, dt=1.0)
            original = track_i.current_detection
            track_i.current_detection = track_i.history[-self.push_frames_back]
            v_past_i = velocity_field.compute_velocity(track_i, dt=1.0)
            track_i.current_detection = original
            
            # Compute angle between vectors
            if np.linalg.norm(v_current_i) > 1e-6 and np.linalg.norm(v_past_i) > 1e-6:
                cos_angle = np.dot(v_current_i, v_past_i) / (np.linalg.norm(v_current_i) * np.linalg.norm(v_past_i))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clip untuk avoid numerical errors
                angle_rad = np.arccos(cos_angle)
                direction_change_i = np.degrees(angle_rad)
        
        # Compute direction change untuk track_j
        if len(track_j.history) >= self.push_frames_back:
            v_current_j = velocity_field.compute_velocity(track_j, dt=1.0)
            original = track_j.current_detection
            track_j.current_detection = track_j.history[-self.push_frames_back]
            v_past_j = velocity_field.compute_velocity(track_j, dt=1.0)
            track_j.current_detection = original
            
            # Compute angle between vectors
            if np.linalg.norm(v_current_j) > 1e-6 and np.linalg.norm(v_past_j) > 1e-6:
                cos_angle = np.dot(v_current_j, v_past_j) / (np.linalg.norm(v_current_j) * np.linalg.norm(v_past_j))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                direction_change_j = np.degrees(angle_rad)
        
        return direction_change_i, direction_change_j
    
    def _check_velocity_mismatch(self, track_i: object, track_j: object,
                                velocity_field: 'VelocityField') -> Tuple[bool, float, float]:
        """
        Check velocity mismatch untuk rear-end collision detection
        
        Prinsip: Rear-end collision terjadi ketika rear vehicle lebih cepat dari front vehicle
        (mendekat dan menabrak dari belakang)
        
        Args:
            track_i: First track
            track_j: Second track
            velocity_field: VelocityField
            
        Returns:
            (is_mismatch, v_i_speed, v_j_speed): 
                is_mismatch: True jika ada velocity mismatch (rear > front)
                v_i_speed: Speed of track_i
                v_j_speed: Speed of track_j
        """
        v_i = velocity_field.compute_velocity(track_i, dt=1.0)
        v_j = velocity_field.compute_velocity(track_j, dt=1.0)

        v_i_speed = np.linalg.norm(v_i)
        v_j_speed = np.linalg.norm(v_j)

        # CRITICAL FIX: Static velocity check (v ≈ 0)
        # Jika kedua velocity ≈ 0 → no velocity mismatch (both stationary)
        STATIC_VELOCITY_THRESHOLD = 0.5  # px/fr (increased from 0.3 for consistency)
        if v_i_speed < STATIC_VELOCITY_THRESHOLD and v_j_speed < STATIC_VELOCITY_THRESHOLD:
            return False, v_i_speed, v_j_speed

        # Check apakah saling mendekat berdasarkan posisi relatif
        bbox_i = track_i.current_detection['bbox']
        bbox_j = track_j.current_detection['bbox']
        
        # Center positions
        center_i = np.array([(bbox_i[0] + bbox_i[2]) / 2, (bbox_i[1] + bbox_i[3]) / 2])
        center_j = np.array([(bbox_j[0] + bbox_j[2]) / 2, (bbox_j[1] + bbox_j[3]) / 2])
        
        # Relative position vector (from i to j)
        rel_pos = center_j - center_i
        rel_pos_norm = np.linalg.norm(rel_pos)
        
        # Direction alignment check: rear-end hanya valid jika kedua kendaraan bergerak
        # searah atau mendekati searah. Papasan/cross-traffic (cos < -0.3) bukan rear-end.
        # Threshold -0.3 dipilih karena:
        #   - Rear-end/same-direction: cos ≈ +1.0 (searah) → aman
        #   - T-bone/perpendicular: cos ≈ 0 → aman (bukan papasan)
        #   - Papasan 180°: cos ≈ -1.0 → diblokir ✓
        #   - Intersection cross ~150°: cos ≈ -0.5 sampai -0.3 → diblokir ✓
        #   - v7 rear-end: cos ≈ +1.0 → aman ✓
        if v_i_speed > 0.5 and v_j_speed > 0.5:
            cos_vv = np.dot(v_i, v_j) / (v_i_speed * v_j_speed)
            same_direction = cos_vv > -0.3
        else:
            same_direction = True  # salah satu hampir diam, tidak bisa hitung arah

        if rel_pos_norm < 1e-6:
            # Overlapping - check velocity difference
            is_mismatch = same_direction and abs(v_i_speed - v_j_speed) > 1.0
        else:
            # Normalize relative position
            rel_pos_unit = rel_pos / rel_pos_norm

            # Project velocities onto relative position direction
            v_i_proj = np.dot(v_i, rel_pos_unit)  # Component of v_i toward j
            v_j_proj = np.dot(v_j, rel_pos_unit)  # Component of v_j toward i (negative if moving away)

            # Velocity mismatch: i approaching j AND faster, ONLY jika searah (bukan papasan)
            is_mismatch = same_direction and (
                (v_i_proj > 0.5 and v_i_speed > v_j_speed) or
                (v_j_proj < -0.5 and v_i_speed > v_j_speed)
            )

        return is_mismatch, v_i_speed, v_j_speed
    
    def _check_approaching(self, track_i: object, track_j: object,
                          velocity_field: 'VelocityField') -> bool:
        """
        Check apakah dua kendaraan saling mendekat (approaching)
        
        Prinsip: Collision terjadi ketika kendaraan saling mendekat
        Traffic jam = kendaraan berdekatan tapi tidak saling mendekat
        
        Args:
            track_i: First track
            track_j: Second track
            velocity_field: VelocityField
            
        Returns:
            True jika saling mendekat, False jika tidak
        """
        # Get velocities first
        v_i = velocity_field.compute_velocity(track_i, dt=1.0)
        v_j = velocity_field.compute_velocity(track_j, dt=1.0)
        v_i_speed = np.linalg.norm(v_i)
        v_j_speed = np.linalg.norm(v_j)

        # CRITICAL FIX: Static velocity check (v ≈ 0)
        # Jika kedua velocity ≈ 0 → NOT approaching (both stationary)
        STATIC_VELOCITY_THRESHOLD = 0.5  # px/fr (increased from 0.3 for consistency)
        if v_i_speed < STATIC_VELOCITY_THRESHOLD and v_j_speed < STATIC_VELOCITY_THRESHOLD:
            return False

        # Get current positions
        bbox_i = track_i.current_detection['bbox']
        bbox_j = track_j.current_detection['bbox']

        center_i = np.array([(bbox_i[0] + bbox_i[2]) / 2, (bbox_i[1] + bbox_i[3]) / 2])
        center_j = np.array([(bbox_j[0] + bbox_j[2]) / 2, (bbox_j[1] + bbox_j[3]) / 2])

        # Relative position vector (from i to j)
        rel_pos = center_j - center_i
        rel_pos_norm = np.linalg.norm(rel_pos)

        if rel_pos_norm < 1e-6:
            # Already overlapping - check if relative velocity indicates collision
            rel_vel = v_i - v_j
            rel_vel_norm = np.linalg.norm(rel_vel)
            # If relative velocity is significant, they were approaching
            return rel_vel_norm > 1.0

        # Normalize relative position
        rel_pos_unit = rel_pos / rel_pos_norm
        
        # Project velocities onto relative position direction
        v_i_proj = np.dot(v_i, rel_pos_unit)  # Component of v_i toward j
        v_j_proj = np.dot(v_j, -rel_pos_unit)  # Component of v_j toward i (negative of rel_pos_unit)
        
        # Approaching: both velocities point toward each other
        # OR one is significantly faster and moving toward the other
        is_approaching = (v_i_proj > 0.5 and v_j_proj > 0.5) or \
                        (v_i_proj > 1.0 and v_i_proj > abs(v_j_proj)) or \
                        (v_j_proj > 1.0 and v_j_proj > abs(v_i_proj))
        
        return is_approaching
    
    def _check_traffic_jam(self, track_i: object, track_j: object,
                          velocity_field: 'VelocityField',
                          energy_loss_i: float, energy_loss_j: float,
                          variance_i: float, variance_j: float) -> bool:
        """
        Check apakah ini traffic jam (mobil merayap berdekatan) atau collision
        
        Prinsip Fisika:
        - Traffic Jam: Velocity rendah SEBELUM overlap, tetap rendah SETELAH overlap (konstan)
        - Collision: Velocity tinggi SEBELUM overlap, drop drastis SETELAH overlap
        
        Args:
            track_i: First track
            track_j: Second track
            velocity_field: VelocityField
            energy_loss_i: Energy loss untuk track_i
            energy_loss_j: Energy loss untuk track_j
            variance_i: Acceleration variance untuk track_i
            variance_j: Acceleration variance untuk track_j
            
        Returns:
            True jika traffic jam, False jika collision
        """
        # Check velocity SEBELUM overlap (5 frames sebelum overlap)
        v_before_i = 0.0
        v_before_j = 0.0
        v_after_i = 0.0
        v_after_j = 0.0
        
        # Get velocity SEBELUM overlap (5 frames sebelum)
        if len(track_i.history) >= self.proximity_frames_back:
            original = track_i.current_detection
            track_i.current_detection = track_i.history[-self.proximity_frames_back]
            v_before_i_vec = velocity_field.compute_velocity(track_i, dt=1.0)
            track_i.current_detection = original
            v_before_i = np.linalg.norm(v_before_i_vec)
        
        if len(track_j.history) >= self.proximity_frames_back:
            original = track_j.current_detection
            track_j.current_detection = track_j.history[-self.proximity_frames_back]
            v_before_j_vec = velocity_field.compute_velocity(track_j, dt=1.0)
            track_j.current_detection = original
            v_before_j = np.linalg.norm(v_before_j_vec)
        
        # Get velocity SETELAH overlap (current)
        v_after_i_vec = velocity_field.compute_velocity(track_i, dt=1.0)
        v_after_j_vec = velocity_field.compute_velocity(track_j, dt=1.0)
        v_after_i = np.linalg.norm(v_after_i_vec)
        v_after_j = np.linalg.norm(v_after_j_vec)
        
        # Compute velocity change ratio
        velocity_change_ratio_i = 0.0
        velocity_change_ratio_j = 0.0
        
        if v_before_i > 1e-6:
            velocity_change_ratio_i = abs(v_after_i - v_before_i) / v_before_i
        
        if v_before_j > 1e-6:
            velocity_change_ratio_j = abs(v_after_j - v_before_j) / v_before_j
        
        # Traffic Jam jika:
        # 1. Velocity rendah SEBELUM overlap (v_before < 2.0 px/frame)
        # 2. Velocity tetap rendah SETELAH overlap (v_after < 2.0 px/frame)
        # 3. Velocity change ratio kecil (< 0.2) - perubahan kecil, konstan
        # 4. Energy loss kecil (< 0.2) - tidak ada energy loss
        # 5. Acceleration variance rendah (< 2.0) - gradual acceleration
        # 
        # ENHANCED: Exception untuk high energy loss (meskipun velocity rendah)
        # High energy loss (> 0.5) = collision indicator, bukan traffic jam
        high_energy_loss = (energy_loss_i > 0.5 or energy_loss_j > 0.5)
        
        is_traffic_jam = (
            v_before_i < self.traffic_jam_velocity_threshold and 
            v_before_j < self.traffic_jam_velocity_threshold and  # Velocity rendah sebelum overlap
            v_after_i < self.traffic_jam_velocity_threshold and 
            v_after_j < self.traffic_jam_velocity_threshold and  # Velocity tetap rendah setelah overlap
            velocity_change_ratio_i < self.traffic_jam_velocity_change_threshold and 
            velocity_change_ratio_j < self.traffic_jam_velocity_change_threshold and  # Perubahan kecil, konstan
            energy_loss_i < self.traffic_jam_energy_loss_threshold and 
            energy_loss_j < self.traffic_jam_energy_loss_threshold and  # Tidak ada energy loss
            variance_i < self.traffic_jam_acceleration_variance_threshold and 
            variance_j < self.traffic_jam_acceleration_variance_threshold and  # Gradual acceleration
            not high_energy_loss  # ENHANCED: Exception untuk high energy loss (collision indicator)
        )
        
        return is_traffic_jam
    
    def _update_overlap_history(self, track_i: object, track_j: object, iou: float, current_frame: int) -> int:
        """
        Update overlap history untuk track berapa lama dua objek overlap
        
        Prinsip:
        - Collision = TRANSIENT event (overlap tiba-tiba, < 5 frames)
        - Traffic Jam = PERSISTENT state (overlap lama, > 10 frames)
        
        Args:
            track_i: First track
            track_j: Second track
            iou: Current IoU value
            current_frame: Current frame number
            
        Returns:
            overlap_duration: Berapa frame overlap sudah terjadi (0 = first frame overlap)
        """
        # Create pair key (sorted untuk consistency)
        pair_key = tuple(sorted([track_i.track_id, track_j.track_id]))
        
        # IoU threshold untuk consider as "overlap" (relaxed)
        overlap_threshold = 0.05  # Very low threshold to catch proximity
        
        if iou > overlap_threshold:
            # Ada overlap - update history
            if pair_key not in self.overlap_history:
                # First overlap frame
                self.overlap_history[pair_key] = {
                    'frames': [current_frame],
                    'last_iou': iou,
                    'last_frame': current_frame
                }
                return 1  # First frame of overlap
            else:
                # Existing overlap - check if continuous
                last_frame = self.overlap_history[pair_key]['last_frame']
                
                # Continuous overlap: frame sekarang = last_frame + 1 (toleransi ±2 frames)
                if abs(current_frame - last_frame) <= 2:
                    # Continuous overlap
                    self.overlap_history[pair_key]['frames'].append(current_frame)
                    self.overlap_history[pair_key]['last_iou'] = iou
                    self.overlap_history[pair_key]['last_frame'] = current_frame
                else:
                    # Overlap baru (reset - ada gap)
                    self.overlap_history[pair_key] = {
                        'frames': [current_frame],
                        'last_iou': iou,
                        'last_frame': current_frame
                    }
                    return 1  # Reset, first frame of new overlap
                
                return len(self.overlap_history[pair_key]['frames'])
        else:
            # Tidak ada overlap - cleanup jika ada di history
            if pair_key in self.overlap_history:
                del self.overlap_history[pair_key]
            return 0
        
    def _check_overlap_duration(self, track_i: object, track_j: object) -> Tuple[int, str]:
        """
        Check overlap duration untuk determine jika transient (collision) atau persistent (traffic jam)
        
        Returns:
            (overlap_duration, state):
                overlap_duration: Jumlah frames overlap (0 = no overlap)
                state: 'transient', 'persistent', or 'normal'
        """
        pair_key = tuple(sorted([track_i.track_id, track_j.track_id]))
        
        if pair_key not in self.overlap_history:
            return 0, 'normal'  # No overlap
        
        overlap_duration = len(self.overlap_history[pair_key]['frames'])
        
        if overlap_duration <= self.transient_overlap_max:
            return overlap_duration, 'transient'  # Collision (< 5 frames)
        elif overlap_duration >= self.persistent_overlap_min:
            return overlap_duration, 'persistent'  # Traffic jam (> 10 frames)
        else:
            return overlap_duration, 'normal'  # In-between (ambiguous)
    
    def _cleanup_overlap_history(self, current_frame: int):
        """
        Cleanup overlap history yang sudah lama (> 100 frames ago)
        
        Args:
            current_frame: Current frame number
        """
        # Only cleanup every N frames to reduce overhead
        if current_frame % self.overlap_history_cleanup_interval != 0:
            return
        
        # Cleanup pairs yang sudah > 100 frames since last update
        pairs_to_remove = []
        for pair_key, data in self.overlap_history.items():
            last_frame = data['last_frame']
            if current_frame - last_frame > 100:
                pairs_to_remove.append(pair_key)
        
        for pair_key in pairs_to_remove:
            del self.overlap_history[pair_key]
        
        if pairs_to_remove:
            logger.debug(f"Cleaned up {len(pairs_to_remove)} old overlap pairs from history")
    
    def _compute_impulse(self, track_i: object, track_j: object, 
                        velocity_field: 'VelocityField') -> Tuple[float, float]:
        """
        Compute impulse (J = Δp = m·Δv) untuk kedua track
        
        Impulse adalah perubahan momentum yang terjadi saat collision.
        J = m·(v_after - v_before)
        
        Args:
            track_i: First track
            track_j: Second track
            velocity_field: VelocityField
            
        Returns:
            (impulse_i, impulse_j): Impulse magnitude untuk kedua track
        """
        # Compute impulse untuk track_i
        impulse_i = 0.0
        if len(track_i.history) >= self.impulse_frames_back:
            v_current_i = velocity_field.compute_velocity(track_i, dt=1.0)
            original = track_i.current_detection
            track_i.current_detection = track_i.history[-self.impulse_frames_back]
            v_past_i = velocity_field.compute_velocity(track_i, dt=1.0)
            track_i.current_detection = original
            
            # Mass approximation dari bbox area
            bbox_i = track_i.current_detection['bbox']
            mass_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
            
            # Impulse: J = m·|Δv|
            delta_v_i = v_current_i - v_past_i
            impulse_i = mass_i * np.linalg.norm(delta_v_i)
        
        # Compute impulse untuk track_j
        impulse_j = 0.0
        if len(track_j.history) >= self.impulse_frames_back:
            v_current_j = velocity_field.compute_velocity(track_j, dt=1.0)
            original = track_j.current_detection
            track_j.current_detection = track_j.history[-self.impulse_frames_back]
            v_past_j = velocity_field.compute_velocity(track_j, dt=1.0)
            track_j.current_detection = original
            
            # Mass approximation dari bbox area
            bbox_j = track_j.current_detection['bbox']
            mass_j = (bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1])
            
            # Impulse: J = m·|Δv|
            delta_v_j = v_current_j - v_past_j
            impulse_j = mass_j * np.linalg.norm(delta_v_j)
        
        return impulse_i, impulse_j
    
    def _compute_acceleration_spike(self, track_i: object, track_j: object,
                                   velocity_field: 'VelocityField') -> Tuple[float, float]:
        """
        Compute acceleration spike untuk kedua track
        
        Acceleration spike terjadi saat collision karena sudden change in velocity.
        a = Δv / Δt
        
        Args:
            track_i: First track
            track_j: Second track
            velocity_field: VelocityField
            
        Returns:
            (acc_spike_i, acc_spike_j): Acceleration spike magnitude untuk kedua track
        """
        # Compute acceleration spike untuk track_i
        acc_spike_i = 0.0
        if len(track_i.history) >= self.acceleration_spike_frames_back:
            v_current_i = velocity_field.compute_velocity(track_i, dt=1.0)
            original = track_i.current_detection
            track_i.current_detection = track_i.history[-self.acceleration_spike_frames_back]
            v_past_i = velocity_field.compute_velocity(track_i, dt=1.0)
            track_i.current_detection = original
            
            # Acceleration: a = Δv / Δt (Δt = frames_back)
            delta_v_i = v_current_i - v_past_i
            acc_spike_i = np.linalg.norm(delta_v_i) / self.acceleration_spike_frames_back
        
        # Compute acceleration spike untuk track_j
        acc_spike_j = 0.0
        if len(track_j.history) >= self.acceleration_spike_frames_back:
            v_current_j = velocity_field.compute_velocity(track_j, dt=1.0)
            original = track_j.current_detection
            track_j.current_detection = track_j.history[-self.acceleration_spike_frames_back]
            v_past_j = velocity_field.compute_velocity(track_j, dt=1.0)
            track_j.current_detection = original
            
            # Acceleration: a = Δv / Δt
            delta_v_j = v_current_j - v_past_j
            acc_spike_j = np.linalg.norm(delta_v_j) / self.acceleration_spike_frames_back
        
        return acc_spike_i, acc_spike_j
    
    def _compute_force_magnitude(self, track_i: object, track_j: object,
                                velocity_field: 'VelocityField') -> Tuple[float, float]:
        """
        Compute force magnitude (F = m·a) untuk kedua track
        
        Force adalah produk dari mass dan acceleration.
        F = m·a = m·(Δv / Δt)
        
        Args:
            track_i: First track
            track_j: Second track
            velocity_field: VelocityField
            
        Returns:
            (force_i, force_j): Force magnitude untuk kedua track
        """
        # Compute force untuk track_i
        force_i = 0.0
        if len(track_i.history) >= self.acceleration_spike_frames_back:
            v_current_i = velocity_field.compute_velocity(track_i, dt=1.0)
            original = track_i.current_detection
            track_i.current_detection = track_i.history[-self.acceleration_spike_frames_back]
            v_past_i = velocity_field.compute_velocity(track_i, dt=1.0)
            track_i.current_detection = original
            
            # Mass approximation
            bbox_i = track_i.current_detection['bbox']
            mass_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
            
            # Acceleration
            delta_v_i = v_current_i - v_past_i
            acc_i = np.linalg.norm(delta_v_i) / self.acceleration_spike_frames_back
            
            # Force: F = m·a
            force_i = mass_i * acc_i
        
        # Compute force untuk track_j
        force_j = 0.0
        if len(track_j.history) >= self.acceleration_spike_frames_back:
            v_current_j = velocity_field.compute_velocity(track_j, dt=1.0)
            original = track_j.current_detection
            track_j.current_detection = track_j.history[-self.acceleration_spike_frames_back]
            v_past_j = velocity_field.compute_velocity(track_j, dt=1.0)
            track_j.current_detection = original
            
            # Mass approximation
            bbox_j = track_j.current_detection['bbox']
            mass_j = (bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1])
            
            # Acceleration
            delta_v_j = v_current_j - v_past_j
            acc_j = np.linalg.norm(delta_v_j) / self.acceleration_spike_frames_back
            
            # Force: F = m·a
            force_j = mass_j * acc_j
        
        return force_i, force_j
    
    def _compute_acceleration_variance(self, track: object, 
                                      velocity_field: 'VelocityField') -> float:
        """
        Compute acceleration variance untuk detect sudden vs gradual stop
        
        Blueprint Equation (Section 2.4.5):
        σ²(a_i) = (1/5) * Σ ||a_i(k) - ā_i||²
        
        High variance = sudden acceleration change (collision)
        Low variance = gradual deceleration (normal braking)
        
        Args:
            track: Track object
            velocity_field: VelocityField
            
        Returns:
            Acceleration variance
        """
        # Use configurable frames_required (default 4, reduced from 6)
        frames_required = getattr(self, 'variance_frames_required', 4)
        if len(track.history) < frames_required:
            return 0.0
        
        # Compute acceleration vectors over recent 5 frames (t-5 to t-1)
        # Blueprint: k from t-5 to t-1
        acceleration_vectors = []
        
        for i in range(-5, 0):
            if len(track.history) + i < 1:
                continue
            
            # Velocity at frame i
            original = track.current_detection
            track.current_detection = track.history[i]
            v_i = velocity_field.compute_velocity(track, dt=1.0)
            track.current_detection = original
            
            # Velocity at frame i-1
            if len(track.history) + i - 1 >= 0:
                track.current_detection = track.history[i-1]
                v_i_minus_1 = velocity_field.compute_velocity(track, dt=1.0)
                track.current_detection = original
                
                # Acceleration vector: a_i(k) = v_i(k) - v_i(k-1)
                a = v_i - v_i_minus_1
                acceleration_vectors.append(a)
        
        if len(acceleration_vectors) < 2:
            return 0.0
        
        # Convert to numpy array for easier computation
        acceleration_vectors = np.array(acceleration_vectors)  # Shape: (n, 2)
        
        # Compute mean acceleration vector: ā_i
        mean_acceleration = np.mean(acceleration_vectors, axis=0)  # Shape: (2,)
        
        # Blueprint: σ²(a_i) = (1/5) * Σ ||a_i(k) - ā_i||²
        # Compute squared norm of difference for each frame
        n_frames = len(acceleration_vectors)
        variance_sum = 0.0
        
        for a_k in acceleration_vectors:
            diff = a_k - mean_acceleration
            squared_norm = np.linalg.norm(diff)**2
            variance_sum += squared_norm
        
        # Blueprint uses fixed divisor 1/5 (not 1/(n-1) like sample variance)
        variance = (1.0 / 5.0) * variance_sum
        
        return variance

    def _update_episode_escalation_gate(
        self,
        pair_key: Tuple[int, int],
        frame_id: int,
        debug_info: Optional[Dict],
        iou: float,
        energy_loss_i: float,
        energy_loss_j: float,
        ars_z_i: float,
        ars_z_j: float,
    ) -> Tuple[bool, Dict]:
        """
        Causal warning-to-escalation gate for PPL episodes.

        Yellow warnings are emitted immediately. Red escalation is raised only
        when the current pair episode accumulates strong ARS while KELR remains
        below the saturated nuisance-warning pattern observed in negative logs.
        """
        if not self.episode_escalation_enabled:
            return False, {}
        if not debug_info or debug_info.get('detection_mode') != 'ppl':
            return False, {}

        kelr_i = float(debug_info.get('ppl_kelr_value_i', energy_loss_i))
        kelr_j = float(debug_info.get('ppl_kelr_value_j', energy_loss_j))
        current_kelr = max(kelr_i, kelr_j)
        current_ars = max(float(ars_z_i), float(ars_z_j))
        current_iou = float(debug_info.get('ppl_iou_now', iou))

        state = self.episode_escalation_state.get(pair_key)
        if state is None or frame_id - state.get('last_frame', frame_id) > self.episode_escalation_gap:
            state = {
                'start_frame': frame_id,
                'last_frame': frame_id,
                'frames': set(),
                'max_ars': current_ars,
                'max_kelr': current_kelr,
                'max_iou': current_iou,
                'upgraded': False,
            }
        else:
            state['last_frame'] = frame_id
            state['max_ars'] = max(state.get('max_ars', 0.0), current_ars)
            state['max_kelr'] = max(state.get('max_kelr', 0.0), current_kelr)
            state['max_iou'] = max(state.get('max_iou', 0.0), current_iou)

        state['frames'].add(frame_id)
        self.episode_escalation_state[pair_key] = state

        should_upgrade = (
            not state.get('upgraded', False)
            and state['max_ars'] >= self.episode_escalation_ars_threshold
            and state['max_kelr'] <= self.episode_escalation_kelr_max
        )
        if should_upgrade:
            state['upgraded'] = True

        info = {
            'episode_start_frame': state['start_frame'],
            'episode_last_frame': state['last_frame'],
            'episode_unique_frames': len(state['frames']),
            'episode_max_ars': state['max_ars'],
            'episode_max_kelr': state['max_kelr'],
            'episode_max_iou': state['max_iou'],
            'episode_escalation_rule': (
                f"ARS>={self.episode_escalation_ars_threshold} "
                f"AND KELR<={self.episode_escalation_kelr_max}"
            ),
        }
        return should_upgrade, info
    
    def _create_collision_detection(self, track_i: object, track_j: object, 
                                    velocity_field: 'VelocityField',
                                    debug_info: Dict = None) -> Optional[Dict]:
        """
        Create collision detection result untuk pair of tracks
        
        Args:
            track_i: First track
            track_j: Second track
            velocity_field: VelocityField
            
        Returns:
            Detection dict atau None
        """
        # Compute collision point (intersection center)
        bbox_i = track_i.current_detection['bbox']
        bbox_j = track_j.current_detection['bbox']
        
        collision_point = self._estimate_collision_point(bbox_i, bbox_j)
        
        # Compute energy losses
        adaptive_threshold = self._get_adaptive_threshold()
        energy_loss_i = self._compute_energy_loss_instant(track_i, velocity_field, adaptive_threshold)
        energy_loss_j = self._compute_energy_loss_instant(track_j, velocity_field, adaptive_threshold)
        
        # IoU
        iou = self._compute_iou(bbox_i, bbox_j)

        # ARS z-scores — computed for ALL detections as supplementary measurement.
        # If PPL already provided them via debug_info, use those; otherwise compute now.
        _ars_zi = float(debug_info.get('ars_zscore_i', float('nan'))) if debug_info else float('nan')
        _ars_zj = float(debug_info.get('ars_zscore_j', float('nan'))) if debug_info else float('nan')
        import math as _math
        if _math.isnan(_ars_zi):
            _, _ars_zi = self.ppl._compute_ARS(track_i)
        if _math.isnan(_ars_zj):
            _, _ars_zj = self.ppl._compute_ARS(track_j)

        # Compute impact direction (delta_v = v_i - v_j)
        v_i = velocity_field.compute_velocity(track_i, dt=1.0)
        v_j = velocity_field.compute_velocity(track_j, dt=1.0)
        impact_direction = v_i - v_j  # Direction of impact
        
        # Compute display bbox untuk visualisasi.
        # Jika dua track overlap (IoU > 0): gunakan INTERSECTION bbox — zona tumbukan
        # yang lebih kecil dan akurat.  Union bbox bisa sangat besar jika salah satu
        # track punya deteksi anomali (e.g. bbox besar/aneh), sehingga overlap secara
        # visual dengan kendaraan lain yang tidak terlibat.
        ix1 = max(bbox_i[0], bbox_j[0])
        iy1 = max(bbox_i[1], bbox_j[1])
        ix2 = min(bbox_i[2], bbox_j[2])
        iy2 = min(bbox_i[3], bbox_j[3])
        if ix1 < ix2 and iy1 < iy2:
            # Ada overlap → intersection adalah titik tumbukan aktual
            union_bbox = [ix1, iy1, ix2, iy2]
        else:
            # Tidak overlap → fallback ke union (untuk near-miss monitoring)
            union_bbox = [
                min(bbox_i[0], bbox_j[0]),
                min(bbox_i[1], bbox_j[1]),
                max(bbox_i[2], bbox_j[2]),
                max(bbox_i[3], bbox_j[3])
            ]

        # Get current state dari collision_pairs
        pair_key = tuple(sorted([track_i.track_id, track_j.track_id]))
        current_state = 'confirmed'
        persist_count = 0
        if pair_key in self.collision_pairs:
            current_state = self.collision_pairs[pair_key]['state']
            persist_count = self.collision_pairs[pair_key]['persist_count']
        
        # Include physics-based info dari debug_info jika available
        detection_mode = 'normal'
        tier = 0
        rotation_spike_i = 0.0
        rotation_spike_j = 0.0
        momentum_transfer_i = 0.0
        momentum_transfer_j = 0.0
        relative_velocity_drop = 0.0
        push_acceleration_i = 0.0
        push_acceleration_j = 0.0
        direction_change_i = 0.0
        direction_change_j = 0.0
        ar_inversion_track = None
        ar_past = 0.0
        ar_current = 0.0

        if debug_info:
            detection_mode = debug_info.get('detection_mode', 'normal')
            tier = debug_info.get('tier', 0)
            rotation_spike_i = debug_info.get('rotation_spike_i', 0.0)
            rotation_spike_j = debug_info.get('rotation_spike_j', 0.0)
            momentum_transfer_i = debug_info.get('momentum_transfer_i', 0.0)
            momentum_transfer_j = debug_info.get('momentum_transfer_j', 0.0)
            relative_velocity_drop = debug_info.get('relative_velocity_drop', 0.0)
            push_acceleration_i = debug_info.get('push_acceleration_i', 0.0)
            push_acceleration_j = debug_info.get('push_acceleration_j', 0.0)
            direction_change_i = debug_info.get('direction_change_i', 0.0)
            direction_change_j = debug_info.get('direction_change_j', 0.0)
            ar_inversion_track = debug_info.get('ar_inversion_track', None)
            ar_past = debug_info.get('ar_past', 0.0)
            ar_current = debug_info.get('ar_current', 0.0)

        # ============================================
        # CONFIDENCE SCORE CALCULATION
        # ============================================
        confidence = 0.0

        # POST-COLLISION STATIC: Special confidence path (TIGHTENED v04-eval)
        # Both tracks are now static but were recently moving → use past speed as evidence
        # Scoring is more conservative: requires stronger evidence to reach high confidence
        if detection_mode == 'post_collision_static' and debug_info:
            # variance_i/j now stores robust_speed (median of top-3) instead of single max
            past_speed_i = debug_info.get('variance_i', 0.0)
            past_speed_j = debug_info.get('variance_j', 0.0)
            max_past_speed = max(past_speed_i, past_speed_j)

            # Evidence 1: IoU Overlap (20%) — raised thresholds
            if iou >= 0.4:
                confidence += 20
            elif iou >= 0.25:
                confidence += 15
            elif iou >= 0.15:
                confidence += 10
            else:
                confidence += 5

            # Evidence 2: Past Speed — robust median (35%) — raised thresholds
            if max_past_speed > 20.0:
                confidence += 35  # Very fast → strong collision evidence
            elif max_past_speed > 15.0:
                confidence += 25
            elif max_past_speed > 10.0:
                confidence += 20
            else:
                confidence += 10  # Just above 8.0 threshold → weak evidence

            # Evidence 3: Both tracks had speed (mutual collision) (15%) — raised thresholds
            if past_speed_i > 8.0 and past_speed_j > 8.0:
                confidence += 15  # Both clearly moving → mutual collision
            elif past_speed_i > 5.0 or past_speed_j > 5.0:
                confidence += 10  # One was moving → hit-and-stop
            else:
                confidence += 0   # Weak evidence — no bonus

            # Evidence 4: Energy loss ratio from debug_info (10%)
            debug_eloss_i = debug_info.get('energy_loss_i', 0.0)
            debug_eloss_j = debug_info.get('energy_loss_j', 0.0)
            if max(debug_eloss_i, debug_eloss_j) > 0.7:
                confidence += 10

            # Override energy_loss for the detection result with past-speed based values
            energy_loss_i = debug_eloss_i
            energy_loss_j = debug_eloss_j

            logger.warning(f"[POST-COLLISION CONFIDENCE] "
                          f"Tracks {track_i.track_id} <-> {track_j.track_id} | "
                          f"IoU: {iou:.3f} | Robust past speed: i={past_speed_i:.1f}, j={past_speed_j:.1f} | "
                          f"Confidence: {confidence}")
        else:
            # NORMAL confidence scoring (existing logic)
            # Evidence 1: IoU Overlap (30% weight)
            if iou >= 0.5:
                confidence += 30  # High confidence (clear overlap)
            elif iou >= 0.3:
                confidence += 25  # Medium-high confidence
            elif iou >= 0.2:
                confidence += 20  # Medium confidence
            else:
                confidence += 10  # Low confidence

            # Evidence 2: Energy Loss (40% weight - more important for regular collision)
            max_energy_loss = max(energy_loss_i, energy_loss_j)
            if max_energy_loss > 0.8:
                confidence += 40  # Very high energy loss
            elif max_energy_loss > 0.5:
                confidence += 30  # High energy loss
            elif max_energy_loss > 0.3:
                confidence += 20  # Medium energy loss
            else:
                confidence += 10  # Low energy loss

            # Evidence 3: Speed Change (20% weight)
            speed_i = np.linalg.norm(v_i)
            speed_j = np.linalg.norm(v_j)
            relative_speed = abs(speed_i - speed_j)
            if relative_speed >= 10.0:
                confidence += 20  # High relative speed
            elif relative_speed >= 5.0:
                confidence += 15  # Medium-high relative speed
            elif relative_speed >= 2.0:
                confidence += 10  # Medium relative speed

            # Evidence 4: Rotation/Evasive/AR Inversion (10% weight)
            has_special_evidence = (rotation_spike_i > 0 or rotation_spike_j > 0 or
                                   ar_inversion_track is not None)
            if has_special_evidence:
                confidence += 10

        # Motorcycle collision bonus
        class_i = track_i.current_detection.get('class_name', 'unknown')
        class_j = track_j.current_detection.get('class_name', 'unknown')
        if 'motorcycle' in class_i.lower() or 'motorcycle' in class_j.lower():
            confidence += 5

        # Clamp to 0-100
        confidence = max(0, min(100, confidence))

        # ── PPL Confidence Floor ─────────────────────────────────────────────
        # Ketika PPL yang deteksi collision (physics-validated), confidence
        # scoring heuristic terlalu konservatif untuk kasus sideswipe (CE+ARS):
        #   energy_loss = 0.0 karena KELR=False → confidence hanya ~20-30%
        # Padahal PPL sudah memvalidasi kontak fisik secara geometrik (CE) +
        # deformasi bbox (ARS). Floor minimum berdasarkan primitif yang fire:
        #   CE+KELR: minimum 70% → emergency (merah) — energy loss confirmed
        #   CE+ARS only: minimum 50% → warning (kuning) — deformasi saja
        if debug_info and debug_info.get('detection_mode') == 'ppl':
            has_kelr = any('KELR' in p for p in debug_info.get('ppl_primitives', []))
            if has_kelr:
                confidence = max(confidence, 70)   # → emergency (merah)
            else:  # CE+ARS only
                confidence = max(confidence, 50)   # → warning (kuning)

        episode_upgraded = False
        episode_info = {}
        if confidence < self.collision_confidence_threshold:
            episode_upgraded, episode_info = self._update_episode_escalation_gate(
                pair_key=pair_key,
                frame_id=getattr(track_i, 'current_frame', 0),
                debug_info=debug_info,
                iou=iou,
                energy_loss_i=energy_loss_i,
                energy_loss_j=energy_loss_j,
                ars_z_i=_ars_zi,
                ars_z_j=_ars_zj,
            )
            if episode_upgraded:
                confidence = max(confidence, self.collision_confidence_threshold)
                logger.warning(
                    f"[EPISODE ESCALATION] Frame {getattr(track_i, 'current_frame', 0)}: "
                    f"Track {track_i.track_id} <-> Track {track_j.track_id} | "
                    f"Rule: {episode_info.get('episode_escalation_rule')} | "
                    f"episode_max_ars={episode_info.get('episode_max_ars', 0.0):.2f} | "
                    f"episode_max_kelr={episode_info.get('episode_max_kelr', 0.0):.3f} | "
                    f"episode_frames={episode_info.get('episode_unique_frames', 0)} | "
                    f"Yellow warning upgraded to red escalation"
                )

        low_fragment_upgraded = False
        low_fragment_tracks_created = int(getattr(self, 'total_tracks_created_so_far', 0) or 0)
        low_fragment_active_tracks = len(tracks) if tracks is not None else 0
        low_fragment_kelr_max = max(float(energy_loss_i), float(energy_loss_j))
        if (
            self.low_fragment_escalation_enabled
            and confidence >= 50
            and confidence < self.collision_confidence_threshold
            and low_fragment_tracks_created <= self.low_fragment_track_threshold
            and low_fragment_active_tracks <= self.low_fragment_active_track_threshold
            and low_fragment_kelr_max >= self.low_fragment_kelr_min
        ):
            confidence = max(confidence, self.collision_confidence_threshold)
            low_fragment_upgraded = True
            logger.warning(
                f"[LOW-FRAGMENT ESCALATION] Frame {getattr(track_i, 'current_frame', 0)}: "
                f"Track {track_i.track_id} <-> Track {track_j.track_id} | "
                f"tracks_created={low_fragment_tracks_created} <= {self.low_fragment_track_threshold} | "
                f"active_tracks={low_fragment_active_tracks} <= {self.low_fragment_active_track_threshold} | "
                f"kelr_max={low_fragment_kelr_max:.3f} >= {self.low_fragment_kelr_min} | "
                f"Critical interaction upgraded to red escalation"
            )

        confidence, scene_guard_promoted, scene_guard_suppressed = self._apply_global_red_scene_guard(
            confidence=confidence,
            tracks=tracks,
            frame_id=getattr(track_i, 'current_frame', 0),
            pair_desc=f"Track {track_i.track_id} <-> Track {track_j.track_id}",
            context='pairwise_collision',
        )

        # Alert level based on confidence (configurable threshold)
        if confidence >= self.collision_confidence_threshold:
            alert_level = 'emergency'
            severity = 'critical'
            label = 'COLLISION DETECTED'
        elif confidence >= 50:
            alert_level = 'warning'
            severity = 'medium'
            label = 'CRITICAL INTERACTION'
        elif confidence >= 30:
            alert_level = 'caution'
            severity = 'low'
            label = 'UNSAFE PROXIMITY'
        else:
            alert_level = 'info'
            severity = 'low'
            label = 'NORMAL'

        return {
            'track_id': track_i.track_id,  # Primary track
            'track_id_secondary': track_j.track_id,
            'frame_id': track_i.current_frame,
            'detector': self.detector_name,
            'behaviour_type': 'collision',
            'bbox': union_bbox,  # Union bbox untuk visualisasi
            'collision_point': collision_point,
            'impact_direction': impact_direction.tolist() if isinstance(impact_direction, np.ndarray) else impact_direction,
            'iou_overlap': float(iou),
            'energy_loss_primary': float(energy_loss_i),
            'energy_loss_secondary': float(energy_loss_j),
            'bbox_primary': bbox_i,
            'bbox_secondary': bbox_j,
            'class_primary': class_i,
            'class_secondary': class_j,
            'severity': severity,
            'alert_level': alert_level,
            'confidence': confidence,  # NEW: Confidence score (0-100)
            'confidence_label': label,  # NEW: Human-readable label
            'metric_value': float(iou),  # Use IoU as metric
            'persistence': 1.0,  # Collision is instant event
            'prediction_mode': 'confirmed',  # Always confirmed untuk collision
            'state': current_state,  # monitoring or confirmed
            'persist_count': persist_count,  # Current persist count
            'low_fragment_escalated': low_fragment_upgraded,
            'low_fragment_tracks_created': low_fragment_tracks_created,
            # Multi-tier detection info
            'detection_mode': detection_mode,  # 'tier1_high_confidence', 'tier1_rotation_spike', 'tier1_5_sparse_high_confidence', etc.
            'tier': tier,  # 1, 1.5, 2, or 3
            # Rotation spike info (NEW - untuk detect sudden spin)
            'rotation_spike_i': float(rotation_spike_i),
            'rotation_spike_j': float(rotation_spike_j),
            # Push collision info (NEW - untuk detect rear-end collision)
            'momentum_transfer_i': float(momentum_transfer_i),
            'momentum_transfer_j': float(momentum_transfer_j),
            'relative_velocity_drop': float(relative_velocity_drop),
            'push_acceleration_i': float(push_acceleration_i),
            'push_acceleration_j': float(push_acceleration_j),
            'direction_change_i': float(direction_change_i),
            'direction_change_j': float(direction_change_j),
            # AR Inversion info (NEW - untuk detect motor jatuh setelah collision)
            'ar_inversion_track': ar_inversion_track,  # Track ID yang mengalami AR inversion
            'ar_past': float(ar_past),  # AR sebelum collision (motor berdiri)
            'ar_current': float(ar_current),  # AR setelah collision (motor jatuh)
            # Legacy physics-based detection info (for backward compatibility)
            'impulse_i': float(debug_info.get('impulse_i', 0.0)) if debug_info else 0.0,
            'impulse_j': float(debug_info.get('impulse_j', 0.0)) if debug_info else 0.0,
            'acc_spike_i': float(debug_info.get('acc_spike_i', 0.0)) if debug_info else 0.0,
            'acc_spike_j': float(debug_info.get('acc_spike_j', 0.0)) if debug_info else 0.0,
            'force_i': float(debug_info.get('force_i', 0.0)) if debug_info else 0.0,
            'force_j': float(debug_info.get('force_j', 0.0)) if debug_info else 0.0,
            # ARS z-scores (PPL primitive, computed for all detections)
            'ars_zscore_i': _ars_zi,
            'ars_zscore_j': _ars_zj,
            # Episode-level escalation gate diagnostics
            'episode_escalated': bool(episode_upgraded),
            'episode_escalation_rule': episode_info.get('episode_escalation_rule', ''),
            'episode_start_frame': episode_info.get('episode_start_frame', 0),
            'episode_unique_frames': episode_info.get('episode_unique_frames', 0),
            'episode_max_ars': float(episode_info.get('episode_max_ars', 0.0)),
            'episode_max_kelr': float(episode_info.get('episode_max_kelr', 0.0)),
            'episode_max_iou': float(episode_info.get('episode_max_iou', 0.0)),
        }
    
    def _estimate_collision_point(self, bbox1: List[float], 
                                  bbox2: List[float]) -> List[float]:
        """
        Estimate collision point dari intersection center
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            [x, y] collision point
        """
        # Intersection bbox
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Center of intersection
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        return [cx, cy]

    def _generate_disappearance_collision(self, disappeared_id: int, other_id: int,
                                         current_frame: int, proximity_data: Dict,
                                         tracks: List, velocity_field: 'VelocityField',
                                         detections: List[Dict]) -> bool:
        """
        Generate collision detection untuk disappearance-based collision

        Ketika track hilang tiba-tiba setelah proximity warning dengan track lain,
        ini kemungkinan besar collision (misal: motor ditabrak mobil lalu hilang)

        Args:
            disappeared_id: ID track yang hilang
            other_id: ID track yang masih aktif
            current_frame: Current frame number
            proximity_data: Data dari proximity_monitoring
            tracks: List of current active tracks
            velocity_field: VelocityField object
            detections: List untuk append collision detection

        Returns:
            bool: True if collision was generated, False if filtered out
        """
        # Get track yang masih aktif
        other_track = next((t for t in tracks if t.track_id == other_id), None)
        if other_track is None:
            return False

        # Get data disappeared track dari last_seen
        disappeared_data = self.track_last_seen.get(disappeared_id)
        if disappeared_data is None:
            return False

        # Create collision detection
        max_iou = proximity_data['max_iou']
        class_disappeared = disappeared_data['class_name']
        class_other = other_track.current_detection.get('class_name', 'unknown')

        # ============================================
        # FILTER: Skip person pairs EXCEPT person+motorcycle
        # ============================================
        has_person = self._is_pedestrian_class(class_disappeared.lower()) or self._is_pedestrian_class(class_other.lower())
        is_person_moto = self._is_person_motorcycle_pair(class_disappeared.lower(), class_other.lower())

        if has_person and not is_person_moto:
            logger.debug(f"⏭️ SKIP _generate_disappearance_collision | "
                        f"Track {disappeared_id} ({class_disappeared}) + Track {other_id} ({class_other}) | "
                        f"Person+vehicle pair - skipping (not person+motorcycle)")
            return False

        # Compute collision point (use last known bbox of disappeared track)
        bbox_disappeared = disappeared_data['bbox']
        bbox_other = other_track.current_detection['bbox']
        collision_point = self._compute_collision_point(bbox_disappeared, bbox_other)

        # Compute display bbox untuk visualisasi (intersection jika overlap, union jika tidak)
        ix1 = max(bbox_disappeared[0], bbox_other[0])
        iy1 = max(bbox_disappeared[1], bbox_other[1])
        ix2 = min(bbox_disappeared[2], bbox_other[2])
        iy2 = min(bbox_disappeared[3], bbox_other[3])
        if ix1 < ix2 and iy1 < iy2:
            union_bbox = [ix1, iy1, ix2, iy2]
        else:
            union_bbox = [
                min(bbox_disappeared[0], bbox_other[0]),
                min(bbox_disappeared[1], bbox_other[1]),
                max(bbox_disappeared[2], bbox_other[2]),
                max(bbox_disappeared[3], bbox_other[3])
            ]

        # ============================================
        # FILTER ID SWITCH: Speed Similarity Check
        # ============================================
        # Jika IoU tinggi + speed hampir sama → likely ID switch (same object), bukan collision
        # Get velocities (reuse computation from motorcycle-person filter above if available)
        disappeared_velocity = 0.0
        other_velocity = 0.0

        # Get disappeared track velocity from motion history
        if disappeared_id in self.track_motion_history:
            velocities = self.track_motion_history[disappeared_id]['velocities']
            if len(velocities) > 0:
                recent_vel = velocities[-1]
                disappeared_velocity = np.linalg.norm(recent_vel)

        # Get other track velocity from motion history
        if other_id in self.track_motion_history:
            velocities = self.track_motion_history[other_id]['velocities']
            if len(velocities) > 0:
                recent_vel = velocities[-1]
                other_velocity = np.linalg.norm(recent_vel)

        # Check speed similarity
        speed_diff = abs(disappeared_velocity - other_velocity)

        # ============================================
        # CONFIDENCE SCORE CALCULATION (Multi-Evidence Scoring)
        # ============================================
        confidence = 0.0

        # Evidence 1: IoU Overlap (30% weight) - FIXED: Consider speed_diff to distinguish ID switch vs real collision
        # High IoU can be EITHER:
        # - ID switch (high IoU + low speed diff)
        # - Real collision (high IoU + high speed diff)
        if max_iou >= 0.7:
            # Check speed_diff to determine if ID switch or real collision
            if speed_diff >= 5.0:
                # High IoU + High speed diff = Real collision with severe overlap
                confidence += 30  # High confidence (severe collision)
            elif speed_diff >= 2.0:
                # High IoU + Medium speed diff = Likely collision
                confidence += 20  # Medium-high confidence
            else:
                # High IoU + Low speed diff = Likely ID switch
                confidence += 5  # Low confidence (likely ID switch)
        elif max_iou >= 0.5:
            confidence += 15  # Medium-low confidence
        elif max_iou >= 0.3:
            confidence += 30  # High confidence (clear collision)
        elif max_iou >= 0.2:
            confidence += 25  # Medium confidence
        else:
            confidence += 10  # Low confidence (weak overlap)

        # Evidence 2: Speed Similarity (25% weight)
        if speed_diff < 2.0:
            confidence -= 15  # Reduce confidence (likely ID switch)
        elif speed_diff >= 8.0:
            confidence += 25  # High confidence (significant impact)
        elif speed_diff >= 5.0:
            confidence += 20  # Medium-high confidence
        elif speed_diff >= 2.0:
            confidence += 15  # Medium confidence

        # Evidence 3: Energy Loss (20% weight) - estimate from speed change
        # For disappeared track: use sudden deceleration as proxy
        energy_loss_estimate = 0.0
        if disappeared_id in self.track_motion_history:
            history = self.track_motion_history[disappeared_id]
            if len(history['velocities']) >= 2:
                v_prev = np.linalg.norm(history['velocities'][-2]) if len(history['velocities']) > 1 else disappeared_velocity
                v_curr = disappeared_velocity
                if v_prev > 0.5:  # Avoid division by zero
                    energy_loss_estimate = abs(v_prev - v_curr) / v_prev

        # FIXED: Also check sudden deceleration percentage (from log: "Drop: X.XX px/fr (XX.X%)")
        # If sudden deceleration detected (speed drop ≥20%), consider as energy loss
        sudden_decel_pct = 0.0
        if disappeared_id in self.track_motion_history:
            history = self.track_motion_history[disappeared_id]
            if len(history['velocities']) >= 2:
                v_prev = np.linalg.norm(history['velocities'][-2]) if len(history['velocities']) > 1 else disappeared_velocity
                v_curr = disappeared_velocity
                if v_prev > 0.5:
                    sudden_decel_pct = abs(v_prev - v_curr) / v_prev

        # Use MAXIMUM of energy_loss_estimate or sudden_decel_pct
        max_energy_indicator = max(energy_loss_estimate, sudden_decel_pct)

        if max_energy_indicator > 0.8:
            confidence += 20  # High confidence
        elif max_energy_indicator > 0.5:
            confidence += 15  # Medium confidence
        elif max_energy_indicator > 0.3:
            confidence += 10  # Low-medium confidence
        elif max_energy_indicator > 0.2:
            confidence += 5  # Low confidence (sudden decel 20-30%)

        # Evidence 4: Evasive Maneuver (15% weight)
        # Check if disappeared track had evasive maneuver
        is_evasive_disappeared = False
        if disappeared_id in self.track_motion_history:
            history = self.track_motion_history[disappeared_id]
            if len(history['velocities']) >= 3:
                last_velocity = history['velocities'][-1]
                is_evasive, _, _, _, _ = self._detect_evasive_maneuver(disappeared_id, last_velocity)
                if is_evasive:
                    confidence += 15
                    is_evasive_disappeared = True

        # Evidence 5: Motorcycle Collision Bonus (Critical vulnerability)
        if 'motorcycle' in class_disappeared.lower() or 'motorcycle' in class_other.lower():
            confidence += 10  # Motorcycle collision more dangerous

        # Clamp confidence to 0-100 range
        confidence = max(0, min(100, confidence))

        # ============================================
        # ALERT LEVEL based on Confidence Score (configurable threshold)
        # ============================================
        # Guard C: Ghost track / speed_diff sanity ─────────────────────────────────
        # speed_diff > 300 px/fr tidak mungkin untuk kendaraan di jalan nyata
        # (300 px/fr @ 30fps = ~900 km/h di kamera tipikal = noise track atau ghost track).
        if speed_diff > 300.0:
            logger.warning(
                f"  ⏭️ SKIP disappearance collision (GHOST TRACK / SPEED SANITY) | "
                f"Track {disappeared_id} <-> {other_id} | "
                f"Speed diff: {speed_diff:.1f} px/fr > 300 — likely noise/ghost track"
            )
            return False
        # ─────────────────────────────────────────────────────────────────────────

        # Guard: Kinetic evidence wajib ada ──────────────────────────────────────
        # Relax: Jika sebelumnya ada proximity yang sangat tinggi (max_iou > 0.3),
        # maka syarat kinetic evidence diperlonggar (kemungkinan besar oklusi karena benturan).
        _has_kinetic_evidence = (
            max_energy_indicator > 0.05 or  # lowered from 0.1
            is_evasive_disappeared or
            max_iou > 0.3                    # proximity override (hit and occluded)
        )
        if not _has_kinetic_evidence:
            logger.warning(
                f"  ⏭️ SKIP disappearance collision (NO KINETIC EVIDENCE) | "
                f"Track {disappeared_id} <-> {other_id} | "
                f"energy={max_energy_indicator:.3f} ≤ 0.05, evasive={is_evasive_disappeared}, max_iou={max_iou:.3f} | "
                f"Hanya proximity — likely oklusi/keluar frame, bukan tabrakan"
            )
            return False
        # ─────────────────────────────────────────────────────────────────────────

        confidence, scene_guard_promoted, scene_guard_suppressed = self._apply_global_red_scene_guard(
            confidence=confidence,
            tracks=None,
            frame_id=current_frame,
            pair_desc=f"Track {other_id} <-> Track {disappeared_id}",
            context='disappearance_collision',
        )

        if confidence >= self.collision_confidence_threshold:
            alert_level = 'emergency'
            severity = 'critical'
            label = 'COLLISION DETECTED'
        elif confidence >= 50:
            alert_level = 'warning'
            severity = 'medium'
            label = 'CRITICAL INTERACTION'
        else:
            # Guard B: Raise minimum confidence 30→50% ────────────────────────────
            # Disappearance FP pattern: confidence 30-45%, energy_loss=0.00, evasive=False.
            # Tidak ada kinetic evidence sama sekali — hanya proximity saat menghilang.
            # Ini bisa dari oklusi, keluar frame, atau ID switch, BUKAN tabrakan.
            # Level "caution" 30-49% dihilangkan — terlalu banyak FP tanpa evidence nyata.
            logger.warning(f"  ⏭️ SKIP disappearance collision (LOW CONFIDENCE) | "
                         f"Track {disappeared_id} <-> {other_id} | "
                         f"Confidence: {confidence:.1f}% < 50% | "
                         f"IoU: {max_iou:.3f}, Speed diff: {speed_diff:.2f} px/fr")
            return False

        # Create detection dict dengan semua field yang diperlukan untuk notification
        detection = {
            'track_id': other_id,
            'track_id_secondary': disappeared_id,
            'behaviour_type': 'collision',
            'severity': severity,
            'alert_level': alert_level,  # Based on confidence score
            'confidence': confidence,  # NEW: Confidence score (0-100)
            'confidence_label': label,  # NEW: Human-readable label
            'iou_overlap': max_iou,
            'collision_point': collision_point,
            'class_primary': class_other,
            'class_secondary': class_disappeared,
            'bbox': union_bbox,  # Union bbox untuk visualisasi
            'bbox_primary': bbox_other,
            'bbox_secondary': bbox_disappeared,
            'energy_loss_primary': 0.0,  # Unknown (track disappeared)
            'energy_loss_secondary': 1.0,  # 100% (track hilang total)
            'energy_loss_estimate': energy_loss_estimate,  # NEW: Estimated from speed change
            'speed_diff': speed_diff,  # NEW: Speed difference evidence
            'is_evasive_disappeared': is_evasive_disappeared,  # NEW: Evasive evidence
            'momentum_transfer_i': 0.0,
            'momentum_transfer_j': 0.0,
            'detection_mode': 'disappearance_collision',
            'tier': 0,  # Disappearance collision tier
            'proximity_frames': len(proximity_data['frames']),
            'max_proximity_iou': max_iou,
            'disappeared_track': disappeared_id,
            'active_track': other_id,
            'frame_id': current_frame,
            'state': 'confirmed',  # Langsung confirmed
            'persist_count': 1,
            'prediction_mode': 'confirmed',
            'persistence': 1.0,  # Collision is instant event
            'metric_value': float(max_iou),  # Use IoU as metric
            'detector': self.detector_name
        }

        detections.append(detection)

        # Log collision with confidence
        logger.warning(f"[DISAPPEARANCE COLLISION] Frame {current_frame}: "
                      f"Track {other_id} ({class_other}) <-> Track {disappeared_id} ({class_disappeared}) | "
                      f"Confidence: {confidence:.1f}% | Alert: {alert_level.upper()} | Label: {label} | "
                      f"IoU: {max_iou:.3f}, Speed diff: {speed_diff:.2f} px/fr | "
                      f"Evidence: energy_loss={energy_loss_estimate:.2f}, evasive={is_evasive_disappeared}")

        # Cleanup proximity monitoring untuk pair ini
        pair_key = tuple(sorted([disappeared_id, other_id]))
        if pair_key in self.proximity_monitoring:
            del self.proximity_monitoring[pair_key]

        return True

    # ============================================
    # EVASIVE MANEUVER DETECTION (NEW!)
    # Deteksi mobil yang tiba-tiba oleng/berubah arah tajam
    # ============================================

    def _update_motion_history(self, track_id: int, velocity: np.ndarray, current_frame: int) -> None:
        """
        Update motion history untuk track (velocity dan direction)

        Args:
            track_id: ID track
            velocity: Velocity vector [vx, vy]
            current_frame: Frame number saat ini
        """
        if track_id not in self.track_motion_history:
            self.track_motion_history[track_id] = {
                'velocities': deque(maxlen=self.motion_history_window),
                'directions': deque(maxlen=self.motion_history_window),
                'frames': deque(maxlen=self.motion_history_window)
            }

        # Always track velocity (termasuk saat berhenti untuk detect deceleration)
        self.track_motion_history[track_id]['velocities'].append(velocity.copy())
        self.track_motion_history[track_id]['frames'].append(current_frame)

        # Hitung direction hanya jika moving (untuk detect evasive maneuver)
        speed = np.linalg.norm(velocity)
        if speed > 0.1:  # Only track direction if moving (lowered from 0.5 to 0.1 for slow traffic)
            direction = np.degrees(np.arctan2(velocity[1], velocity[0]))  # -180 to 180
            self.track_motion_history[track_id]['directions'].append(direction)
        else:
            # Jika tidak bergerak, gunakan direction terakhir atau 0
            if len(self.track_motion_history[track_id]['directions']) > 0:
                last_direction = self.track_motion_history[track_id]['directions'][-1]
                self.track_motion_history[track_id]['directions'].append(last_direction)
            else:
                self.track_motion_history[track_id]['directions'].append(0.0)

    def _detect_evasive_maneuver(self, track_id: int, current_velocity: np.ndarray) -> Tuple[bool, float, float, float, float]:
        """
        Deteksi evasive maneuver (mobil tiba-tiba oleng/berubah arah tajam ATAU sudden stop)

        Args:
            track_id: ID track yang akan dicek
            current_velocity: Current velocity vector [vx, vy]

        Returns:
            Tuple[bool, float, float, float, float]:
                - is_evasive: True jika terdeteksi evasive maneuver OR sudden deceleration
                - angular_velocity: Kecepatan sudut (degree/frame)
                - lateral_accel: Percepatan lateral (px/fr²)
                - direction_change: Total perubahan arah (degree)
                - sudden_deceleration_percentage: Percentage speed drop (0.0-100.0), 0.0 if no deceleration
        """
        # Check if we have enough motion history
        if track_id not in self.track_motion_history:
            return False, 0.0, 0.0, 0.0, 0.0

        history = self.track_motion_history[track_id]
        if len(history['directions']) < 3:  # Need at least 3 frames
            return False, 0.0, 0.0, 0.0, 0.0

        # Check current speed
        current_speed = np.linalg.norm(current_velocity)

        # Calculate direction changes
        directions = list(history['directions'])
        velocities = list(history['velocities'])
        frames = list(history['frames'])

        # Calculate angular velocity (degree/frame) and deceleration
        if len(directions) >= 3 and len(velocities) >= 3:
            # Total direction change over window
            total_direction_change = abs(self._angle_difference(directions[-1], directions[0]))

            # Angular velocity = total change / window duration
            # Pakai window, bukan last-2-frame, karena ketika kendaraan berhenti
            # (speed < 0.1) direction di-repeat → delta last-2 selalu 0 meski ada
            # perubahan nyata di window.
            window_frames = frames[-1] - frames[0]
            if window_frames > 0:
                angular_velocity = total_direction_change / window_frames
            else:
                angular_velocity = total_direction_change / max(len(directions) - 1, 1)

            # Calculate lateral acceleration
            # a_lateral = v * ω (where ω is angular velocity in radians/frame)
            omega_rad = np.radians(angular_velocity)
            lateral_accel = current_speed * omega_rad

            # TAMBAHAN: Calculate deceleration (sudden stop)
            # Compare speed 3 frames ago vs now
            speed_previous = np.linalg.norm(velocities[-3])  # 3 frames ago
            speed_change = speed_previous - current_speed

            # Calculate sudden deceleration percentage
            if speed_previous > 1e-6:  # Avoid division by zero
                sudden_deceleration_percentage = 100.0 * (speed_change / speed_previous)
            else:
                sudden_deceleration_percentage = 0.0

            # Deceleration threshold: jika kecepatan turun > 2.0 px/fr dalam 3 frame
            # atau kecepatan turun > 50% dari kecepatan awal
            sudden_deceleration = (
                speed_change > 2.0 or  # Absolute deceleration > 2 px/fr
                (speed_previous > 1.0 and sudden_deceleration_percentage > 50.0)  # Relative deceleration > 50%
            )

            # Check thresholds - EVASIVE if:
            # 1. Angular velocity tinggi (oleng)
            # 2. Lateral acceleration tinggi (belok tajam)
            # 3. Sudden deceleration (berhenti mendadak) - BARU!
            is_evasive = (
                angular_velocity >= self.evasive_angular_velocity_threshold or
                lateral_accel >= self.evasive_lateral_accel_threshold or
                sudden_deceleration
            )

            if sudden_deceleration:
                logger.warning(f"🛑 SUDDEN DECELERATION | Track {track_id} | "
                              f"Speed: {speed_previous:.2f} → {current_speed:.2f} px/fr | "
                              f"Drop: {speed_change:.2f} px/fr ({sudden_deceleration_percentage:.1f}%)")

            return is_evasive, angular_velocity, lateral_accel, total_direction_change, sudden_deceleration_percentage

        return False, 0.0, 0.0, 0.0, 0.0

    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """
        Calculate smallest difference between two angles (in degrees)
        Result is in range [-180, 180]
        """
        diff = angle1 - angle2
        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def _calculate_victim_velocity_change(self, track_id: int, current_frame: int) -> float:
        """
        Calculate victim's velocity change percentage (untuk detect impact)

        Args:
            track_id: Track ID victim
            current_frame: Frame saat ini

        Returns:
            Velocity change percentage (0.0-1.0)
        """
        if track_id not in self.track_motion_history:
            return 0.0

        history = self.track_motion_history[track_id]
        if len(history['velocities']) < 3:
            return 0.0

        velocities = list(history['velocities'])

        # Compare recent velocity (last 1-2 frames) vs baseline (first 2-3 frames)
        baseline_velocities = velocities[:min(3, len(velocities)//2)]
        recent_velocities = velocities[-min(2, len(velocities)//2):]

        baseline_speed = np.mean([np.linalg.norm(v) for v in baseline_velocities])
        recent_speed = np.mean([np.linalg.norm(v) for v in recent_velocities])

        if baseline_speed < 0.5:  # Too slow to measure change
            return 0.0

        velocity_change = abs(recent_speed - baseline_speed) / baseline_speed
        return min(velocity_change, 1.0)  # Cap at 100%

    def _calculate_victim_direction_change(self, track_id: int, current_frame: int) -> float:
        """
        Calculate victim's direction change (untuk detect impact)

        Args:
            track_id: Track ID victim
            current_frame: Frame saat ini

        Returns:
            Direction change in degrees
        """
        if track_id not in self.track_motion_history:
            return 0.0

        history = self.track_motion_history[track_id]
        if len(history['directions']) < 3:
            return 0.0

        directions = list(history['directions'])

        # Compare recent direction vs baseline
        baseline_direction = directions[0]
        recent_direction = directions[-1]

        direction_change = abs(self._angle_difference(recent_direction, baseline_direction))
        return direction_change

    def _calculate_closing_rate(self, aggressor_track: object, victim_track: object,
                                velocity_field: 'VelocityField') -> float:
        """
        Calculate closing rate between aggressor and victim

        Args:
            aggressor_track: Aggressor track object
            victim_track: Victim track object (or None if disappeared)
            velocity_field: VelocityField instance

        Returns:
            Closing rate in px/frame
        """
        v_aggressor = velocity_field.compute_velocity(aggressor_track, dt=1.0)
        aggressor_speed = np.linalg.norm(v_aggressor)

        if victim_track is None:
            return aggressor_speed  # Victim disappeared, use aggressor speed

        v_victim = velocity_field.compute_velocity(victim_track, dt=1.0)
        victim_speed = np.linalg.norm(v_victim)

        # Closing rate = relative speed approaching each other
        closing_rate = abs(aggressor_speed - victim_speed)
        return closing_rate

    def _calculate_approach_angle(self, aggressor_track: object, victim_track: object,
                                  velocity_field: 'VelocityField') -> float:
        """
        Calculate approach angle between aggressor and victim velocity vectors

        Args:
            aggressor_track: Aggressor track object
            victim_track: Victim track object (or None if disappeared)
            velocity_field: VelocityField instance

        Returns:
            Approach angle in degrees (0-180)
        """
        v_aggressor = velocity_field.compute_velocity(aggressor_track, dt=1.0)

        if victim_track is None:
            return 0.0  # Can't calculate without victim

        v_victim = velocity_field.compute_velocity(victim_track, dt=1.0)

        aggressor_speed = np.linalg.norm(v_aggressor)
        victim_speed = np.linalg.norm(v_victim)

        if aggressor_speed < 0.5 or victim_speed < 0.5:
            return 0.0

        # Normalize vectors
        v_aggressor_unit = v_aggressor / aggressor_speed
        v_victim_unit = v_victim / victim_speed

        # Calculate angle between vectors
        dot_product = np.clip(np.dot(v_aggressor_unit, v_victim_unit), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def _calculate_collision_confidence(self, aggressor_track: object, victim_track_id: int,
                                        victim_track: Optional[object], proximity_data: Dict,
                                        evasive_data: Dict, victim_disappeared: bool,
                                        velocity_field: 'VelocityField', current_frame: int) -> Tuple[float, Dict]:
        """
        Calculate collision confidence score (0-100) based on multiple evidence

        Args:
            aggressor_track: Aggressor track object
            victim_track_id: Victim track ID
            victim_track: Victim track object (or None if disappeared)
            proximity_data: Proximity history data
            evasive_data: Evasive maneuver data
            victim_disappeared: Whether victim disappeared
            velocity_field: VelocityField instance
            current_frame: Current frame number

        Returns:
            Tuple[float, Dict]: (confidence_score, evidence_breakdown)
        """
        confidence = 0.0
        evidence = {}

        # ============================================
        # 1. Proximity Evidence (0-30 points)
        # ============================================
        max_iou = proximity_data.get('max_iou', 0.0)
        proximity_frames = len(proximity_data.get('frames', []))

        if max_iou > 0.4:
            proximity_score = 30
        elif max_iou > 0.3:
            proximity_score = 20
        elif max_iou > 0.2:
            proximity_score = 10
        else:
            proximity_score = 0

        confidence += proximity_score
        evidence['proximity_score'] = proximity_score
        evidence['max_iou'] = max_iou
        evidence['proximity_frames'] = proximity_frames

        # ============================================
        # 2. Evasive Maneuver Evidence (0-25 points)
        # ============================================
        angular_vel = evasive_data.get('angular_velocity', 0.0)
        lateral_acc = evasive_data.get('lateral_acceleration', 0.0)
        aggressor_speed = evasive_data.get('aggressor_speed', 0.0)

        evasive_score = 0
        
        # IMPORTANT: Validate angular velocity against speed to prevent false positives
        # High angular velocity at low speed is likely noise (vehicle stopping/idling)
        # Only consider angular velocity as collision evidence if:
        # 1. Speed is significant (>= 1.0 px/fr) OR
        # 2. Lateral acceleration is high (>10 px/fr²) - indicating actual evasive maneuver
        
        is_high_angular_vel = (angular_vel > 100.0)
        is_low_speed = (aggressor_speed < 1.0)
        is_low_lateral_acc = (lateral_acc < 10.0)
        
        # If high angular velocity at low speed with low lateral accel, this is likely noise
        if is_high_angular_vel and is_low_speed and is_low_lateral_acc:
            # Ignore angular velocity scoring (set to 0) - this is noise, not collision evidence
            logger.warning(f"⚠️ IGNORED HIGH ANGULAR VELOCITY (NOISE) | "
                          f"Angular vel: {angular_vel:.1f}°/fr at speed: {aggressor_speed:.2f} px/fr "
                          f"(< 1.0 px/fr) with lateral acc: {lateral_acc:.2f} px/fr² (< 10.0) | "
                          f"Likely noise from vehicle stopping/idling - not collision evidence")
            angular_vel_score = 0
        else:
            # Enhanced scoring untuk extreme evasive maneuvers (validated)
            if angular_vel > 150:
                angular_vel_score = 25  # Extreme evasive (max score)
            elif angular_vel > 100:
                angular_vel_score = 20  # Very high evasive
            elif angular_vel > 50:
                angular_vel_score = 18  # High evasive
            elif angular_vel > 20:
                angular_vel_score = 15
            elif angular_vel > 15:
                angular_vel_score = 10
            elif angular_vel > 10:
                angular_vel_score = 5
            else:
                angular_vel_score = 0
            
            evasive_score += angular_vel_score

        # Enhanced scoring untuk extreme lateral acceleration
        if lateral_acc > 10:
            evasive_score += 10
        elif lateral_acc > 5:
            evasive_score += 8  # High lateral accel (untuk detect collision seperti Track 107)
        elif lateral_acc > 3:
            evasive_score += 5

        confidence += evasive_score
        evidence['evasive_score'] = evasive_score
        evidence['angular_velocity'] = angular_vel
        evidence['lateral_acceleration'] = lateral_acc

        # ============================================
        # 3. Victim Impact Evidence (0-30 points)
        # ============================================
        impact_score = 0

        if victim_disappeared:
            impact_score = 30
            evidence['victim_status'] = 'disappeared'
        else:
            # Check victim velocity/direction change
            victim_velocity_change = self._calculate_victim_velocity_change(victim_track_id, current_frame)
            victim_direction_change = self._calculate_victim_direction_change(victim_track_id, current_frame)

            if victim_velocity_change > 0.5:  # 50% change
                impact_score += 20
            elif victim_velocity_change > 0.3:  # 30% change
                impact_score += 10

            if victim_direction_change > 30:  # 30 degree
                impact_score += 15
            elif victim_direction_change > 20:  # 20 degree
                impact_score += 10
            elif victim_direction_change > 10:  # 10 degree
                impact_score += 5

            evidence['victim_status'] = 'visible'
            evidence['victim_velocity_change'] = victim_velocity_change
            evidence['victim_direction_change'] = victim_direction_change

        confidence += impact_score
        evidence['impact_score'] = impact_score

        # ============================================
        # 4. Relative Motion Evidence (0-15 points)
        # ============================================
        closing_rate = self._calculate_closing_rate(aggressor_track, victim_track, velocity_field)
        approach_angle = self._calculate_approach_angle(aggressor_track, victim_track, velocity_field)

        motion_score = 0
        if closing_rate > 7:
            motion_score += 10
        elif closing_rate > 5:
            motion_score += 5

        if approach_angle < 30:
            motion_score += 5
        elif approach_angle < 45:
            motion_score += 3

        confidence += motion_score
        evidence['motion_score'] = motion_score
        evidence['closing_rate'] = closing_rate
        evidence['approach_angle'] = approach_angle

        # ============================================
        # Total Confidence (capped at 100)
        # ============================================
        confidence = min(confidence, 100.0)
        evidence['total_confidence'] = confidence

        return confidence, evidence

    def _was_in_proximity(self, track_id_1: int, track_id_2: int, frames: int = 5) -> Tuple[bool, Optional[Dict]]:
        """
        Check apakah 2 tracks pernah ada di proximity dalam N frame terakhir

        Args:
            track_id_1: ID track pertama
            track_id_2: ID track kedua
            frames: Berapa frame ke belakang untuk cek

        Returns:
            Tuple[bool, Optional[Dict]]: (was_in_proximity, proximity_data)
        """
        pair_key = tuple(sorted([track_id_1, track_id_2]))

        if pair_key in self.proximity_monitoring:
            proximity_data = self.proximity_monitoring[pair_key]
            # Check if proximity was within last N frames
            if len(proximity_data['frames']) > 0:
                return True, proximity_data

        return False, None

    def _check_evasive_collision(self, tracks: List, velocity_field: 'VelocityField',
                                 current_frame: int, detections: List[Dict]) -> None:
        """
        Check untuk evasive collision dengan CONFIDENCE SCORING:
        - Mobil yang tiba-tiba oleng (evasive maneuver)
        - Check SEMUA tracks dalam proximity (disappeared OR still visible)
        - Calculate confidence score (0-100)
        - Report jika confidence >= 50 (medium or high)

        Args:
            tracks: List of active tracks
            velocity_field: VelocityField instance
            current_frame: Current frame number
            detections: List untuk append collision detection results
        """
        # Update recently_disappeared - cleanup old entries
        to_remove = []
        for disappeared_id, data in self.recently_disappeared.items():
            frames_since_disappearance = current_frame - data['disappeared_frame']
            if frames_since_disappearance > self.disappeared_retention_frames:
                to_remove.append(disappeared_id)

        for disappeared_id in to_remove:
            del self.recently_disappeared[disappeared_id]

        # Update motion history untuk SEMUA tracks (termasuk motor/person untuk detect impact)
        for track in tracks:
            if not self._is_valid_track(track):
                continue
            velocity = velocity_field.compute_velocity(track, dt=1.0)
            self._update_motion_history(track.track_id, velocity, current_frame)

        # Debug: Log all active tracks and their classes
        logger.debug(f"[EVASIVE COLLISION CHECK] Frame {current_frame} | Active tracks: {len(tracks)}")
        for track in tracks:
            if self._is_valid_track(track):
                class_name = track.current_detection.get('class_name', 'unknown')
                is_vehicle = self._is_vehicle(track)
                logger.debug(f"  Track {track.track_id}: {class_name} | Is vehicle: {is_vehicle}")

        # KELR gate: pre-scan semua tracks untuk PPL physics evidence.
        # Evasive path hanya aktif untuk track yang PPL flagging punya KELR >= threshold.
        # Normal stop (traffic light, gradual decel) → KELR Z-score rendah → tidak masuk.
        # Collision stop (sudden dari kecepatan stabil) → Z-score tinggi → masuk.
        kelr_flagged = {}
        for _t in tracks:
            if not self._is_valid_track(_t):
                continue
            if not self._is_vehicle(_t):
                continue
            if _t.current_detection.get('class_name', '').lower() in ('truck', 'bus'):
                continue
            _k_fires, _k_val = self.ppl._compute_KELR_personal(_t)
            if _k_fires:
                kelr_flagged[_t.track_id] = _k_val
                logger.debug(f"[KELR GATE] Track {_t.track_id} KELR={_k_val:.3f} → evasive candidate")

        # Loop semua active tracks untuk detect evasive maneuver (hanya vehicles)
        for track in tracks:
            if not self._is_valid_track(track):
                continue

            track_id = track.track_id
            class_name = track.current_detection.get('class_name', 'unknown')

            # Only check vehicles (not person/motorcycle for evasive)
            if not self._is_vehicle(track):
                logger.debug(f"[EVASIVE SKIP] Track {track_id} ({class_name}) - not a vehicle")
                continue

            # TRUCK/BUS GUARD: evasive collision tidak aktif untuk truck/bus.
            # Yield/brake behavior saat berpapasan dengan truck besar identik secara fisik
            # dengan post-collision evasive response → FP.
            if class_name.lower() in ('truck', 'bus'):
                logger.debug(f"[EVASIVE SKIP] Track {track_id} ({class_name}) - truck/bus excluded from evasive")
                continue

            # Detect evasive maneuver
            velocity = velocity_field.compute_velocity(track, dt=1.0)
            speed = np.linalg.norm(velocity)

            logger.debug(f"[EVASIVE VEHICLE] Track {track_id} ({class_name}) | Speed: {speed:.2f} px/fr (min: {self.evasive_min_speed})")
            is_evasive, angular_vel, lateral_acc, direction_change, sudden_dec_pct = self._detect_evasive_maneuver(
                track_id, velocity
            )

            # Debug: log evasive check for vehicles (even if not evasive)
            if speed > self.evasive_min_speed and track_id in self.track_motion_history:
                history = self.track_motion_history[track_id]
                if len(history['directions']) >= 2:
                    logger.debug(f"[EVASIVE CHECK] Track {track_id} | "
                               f"Speed: {speed:.2f} px/fr | "
                               f"Angular vel: {angular_vel:.2f}°/fr (threshold: {self.evasive_angular_velocity_threshold}) | "
                               f"Lateral acc: {lateral_acc:.2f} px/fr² (threshold: {self.evasive_lateral_accel_threshold}) | "
                               f"Is evasive: {is_evasive}")

            # KELR gate: jika KELR PPL fires tapi is_evasive False, tetap masuk evasive path.
            # Track yang terhenti mendadak karena impact mungkin tidak menunjukkan angular vel
            # tinggi (lurus berhenti), tapi PPL sudah validasi dengan physics (Z-score guard).
            if not is_evasive and track_id in kelr_flagged:
                is_evasive = True
                logger.warning(f"🔑 KELR GATE OVERRIDE | Track {track_id} | "
                               f"KELR={kelr_flagged[track_id]:.3f} → force evasive entry (no behavioral signal)")

            if is_evasive:
                # JITTER GUARD: direction_change ≥ 150° + lateral_acc < 0.5 px/fr² = YOLO centroid bounce.
                # YOLO bbox centroid pada kendaraan berhenti/lambat berosilasi antara 2 titik →
                # velocity direction flip 180° setiap frame → angular_vel mencapai maximum (45°/fr).
                # Manuver evasive nyata SELALU punya lateral acceleration karena perubahan lintasan fisik.
                # Guard ini tidak suppress track dengan lateral_acc signifikan (real evasive turn).
                # Jitter override: KELR bisa fires pada centroid oscillation (kendaraan berhenti,
                # centroid bouncing antara 2 posisi). Jika KELR fires tapi direction_change ≥ 150°
                # dan lateral_acc < 0.5, ini oscillation, bukan manuver nyata.
                # Exception: lateral_acc ≥ 0.5 = ada gaya fisik nyata → bukan pure jitter.
                if direction_change >= 150.0 and lateral_acc < 0.5:
                    logger.warning(f"⏭️ SKIP EVASIVE (JITTER) | Track {track_id} | "
                                   f"direction_change={direction_change:.1f}° ≥ 150° + "
                                   f"lateral_acc={lateral_acc:.2f} < 0.5 px/fr² → centroid bounce")
                    continue

                logger.warning(f"🔄 EVASIVE MANEUVER DETECTED | Track {track_id} | "
                              f"Angular velocity: {angular_vel:.1f}°/fr | "
                              f"Lateral accel: {lateral_acc:.2f} px/fr² | "
                              f"Direction change: {direction_change:.1f}°")

                # Compute past speed (before deceleration) for collision evidence checks
                aggressor_current_speed = np.linalg.norm(velocity)
                aggressor_past_speed = aggressor_current_speed
                if sudden_dec_pct > 50.0 and track_id in self.track_motion_history:
                    hist_vels = list(self.track_motion_history[track_id]['velocities'])
                    if len(hist_vels) >= 3:
                        aggressor_past_speed = max(aggressor_current_speed, np.linalg.norm(hist_vels[-3]))

                evasive_data = {
                    'angular_velocity': angular_vel,
                    'lateral_acceleration': lateral_acc,
                    'direction_change': direction_change,
                    'aggressor_speed': aggressor_current_speed,
                    'aggressor_past_speed': aggressor_past_speed,
                    'sudden_deceleration_percentage': sudden_dec_pct
                }

                # ============================================
                # NORMAL TURN FILTER: Skip collision detection jika ini hanya belok normal
                # Belok normal = angular velocity < threshold DAN lateral accel < threshold
                # Hanya evasive extreme yang boleh trigger collision detection
                # 
                # EXCEPTION: Jangan skip jika ada collision evidence:
                # 1. Sudden deceleration >50% + direction change >90° (kendaraan terpental akibat collision)
                # 2. Sudden deceleration >50% + ada disappeared track dalam proximity (disappearance collision)
                # ============================================
                
                # Check for collision evidence: sudden deceleration >50% + direction change >90°
                # IMPORTANT: Ignore direction change if speed is too low (< 1.0 px/fr) because
                # direction calculation becomes unreliable at very low speeds (noise/angle wrapping)
                #
                # FIX: Use PAST speed (before deceleration) not current speed.
                # After collision, current speed = 0 but that doesn't mean the collision was fake.
                # A vehicle that went from 19 px/fr → 0 clearly had significant speed.
                past_speed = evasive_data.get('aggressor_speed', 0.0)
                if sudden_dec_pct > 50.0 and track_id in self.track_motion_history:
                    hist_velocities = list(self.track_motion_history[track_id]['velocities'])
                    if len(hist_velocities) >= 3:
                        past_speed = max(past_speed, np.linalg.norm(hist_velocities[-3]))

                # KELR gate menggantikan behavioral decel check.
                # PPL _compute_KELR_personal sudah punya Z-score guard:
                # kendaraan yang berhenti dari kecepatan STABIL (collision impact) → Z-score tinggi → fires.
                # kendaraan yang melambat perlahan (traffic light, gradual) → std_before tinggi → Z-score rendah → tidak fires.
                has_collision_evidence = track_id in kelr_flagged
                
                # Check for disappearance collision evidence: sudden deceleration >50% + disappeared track in proximity
                has_disappearance_collision_evidence = False
                if sudden_dec_pct > 50.0:
                    # Check if there are any disappeared tracks that were in proximity
                    for disappeared_id, disappeared_data in list(self.recently_disappeared.items()):
                        frames_since_disappearance = current_frame - disappeared_data['disappeared_frame']
                        if frames_since_disappearance <= 10:
                            was_in_proximity, _ = self._was_in_proximity(
                                track_id, disappeared_id, frames=self.motion_history_window
                            )
                            if was_in_proximity:
                                has_disappearance_collision_evidence = True
                                break
                
                is_normal_turn = (
                    angular_vel < self.normal_turn_angular_velocity_max and
                    lateral_acc < self.normal_turn_lateral_accel_max and
                    not has_collision_evidence and  # Don't skip if collision evidence exists
                    not has_disappearance_collision_evidence  # Don't skip if disappearance collision evidence exists
                )
                
                if is_normal_turn:
                    logger.warning(f"⏭️ SKIP EVASIVE COLLISION | Track {track_id} | "
                                  f"Angular: {angular_vel:.1f}°/fr < {self.normal_turn_angular_velocity_max}° AND "
                                  f"Lateral: {lateral_acc:.2f} px/fr² < {self.normal_turn_lateral_accel_max} | "
                                  f"Normal turn detected (not extreme evasive) - skipping collision detection")
                    continue  # Skip evasive collision detection untuk belok normal
                
                if has_collision_evidence:
                    logger.warning(f"🔥 KELR COLLISION EVIDENCE | Track {track_id} | "
                                  f"KELR={kelr_flagged.get(track_id, 0):.3f} (PPL physics gate) | "
                                  f"Overriding normal turn filter - proceeding with collision detection")
                
                if has_disappearance_collision_evidence:
                    logger.warning(f"🔥 DISAPPEARANCE COLLISION EVIDENCE DETECTED | Track {track_id} | "
                                  f"Sudden deceleration: {sudden_dec_pct:.1f}% + "
                                  f"Disappeared track in proximity | "
                                  f"Overriding normal turn filter - proceeding with collision detection")

                # ============================================
                # Check 1: Disappeared tracks (highest priority)
                # Hanya kirim notifikasi jika evasive EXTREME + ada objek di proximity
                # ============================================
                logger.debug(f"[EVASIVE] Check disappeared tracks for aggressor {track_id} | "
                           f"Recently disappeared: {len(self.recently_disappeared)}")
                for disappeared_id, disappeared_data in list(self.recently_disappeared.items()):
                    frames_since_disappearance = current_frame - disappeared_data['disappeared_frame']
                    logger.debug(f"[EVASIVE] Checking disappeared {disappeared_id} | "
                               f"Frames since: {frames_since_disappearance}")

                    # ============================================
                    # FILTER 1: Skip jika velocity = 0.0 AND no sudden deceleration
                    # Velocity 0.0 with no prior speed = truly stationary (FP)
                    # Velocity 0.0 with sudden deceleration = just stopped from collision (valid!)
                    # ============================================
                    aggressor_speed = evasive_data.get('aggressor_speed', 0.0)
                    aggressor_past_spd = evasive_data.get('aggressor_past_speed', 0.0)
                    sudden_dec = evasive_data.get('sudden_deceleration_percentage', 0.0)
                    if aggressor_speed == 0.0 and aggressor_past_spd < 1.0 and sudden_dec < 50.0:
                        logger.debug(f"[EVASIVE SKIP] Track {track_id} | "
                                    f"Velocity = 0.0, past_speed = {aggressor_past_spd:.1f}, no sudden decel | "
                                    f"Skipping disappearance collision with Track {disappeared_id} - "
                                    f"truly stationary (not post-collision)")
                        continue  # Skip — truly stationary, not post-collision

                    # ============================================
                    # FILTER 2: Skip person EXCEPT person+motorcycle
                    # ============================================
                    disappeared_class = disappeared_data.get('class_name', 'unknown').lower()
                    aggressor_class = track.current_detection.get('class_name', 'unknown').lower()
                    if self._is_pedestrian_class(disappeared_class) and not self._is_person_motorcycle_pair(disappeared_class, aggressor_class):
                        logger.debug(f"[EVASIVE SKIP] Disappeared Track {disappeared_id} ({disappeared_class}) is person + aggressor ({aggressor_class}) not motorcycle - skipping")
                        continue

                    # Only consider recent disappearances (within 10 frames for slow-moving traffic)
                    if frames_since_disappearance <= 10:
                        # Check if they were in proximity
                        was_in_proximity, proximity_data = self._was_in_proximity(
                            track_id, disappeared_id, frames=self.motion_history_window
                        )

                        if was_in_proximity:
                            # For disappearance collision: Override normal turn filter if sudden deceleration >50%
                            # This is strong collision evidence (vehicle stops suddenly after hitting something)
                            sudden_dec_pct_for_disappearance = evasive_data.get('sudden_deceleration_percentage', 0.0)
                            has_sudden_decel_for_disappearance = sudden_dec_pct_for_disappearance > 50.0
                            
                            if has_sudden_decel_for_disappearance:
                                logger.warning(f"🔥 DISAPPEARANCE COLLISION EVIDENCE | Track {track_id} | "
                                             f"Sudden deceleration: {sudden_dec_pct_for_disappearance:.1f}% after "
                                             f"Track {disappeared_id} disappeared | "
                                             f"Overriding normal turn filter - proceeding with collision detection")
                            
                            # Calculate confidence (victim disappeared = high confidence)
                            confidence, evidence = self._calculate_collision_confidence(
                                aggressor_track=track,
                                victim_track_id=disappeared_id,
                                victim_track=None,  # Disappeared
                                proximity_data=proximity_data,
                                evasive_data=evasive_data,
                                victim_disappeared=True,
                                velocity_field=velocity_field,
                                current_frame=current_frame
                            )

                            logger.warning(f"⚠️ EVASIVE COLLISION (DISAPPEARED) | "
                                         f"Track {track_id} <-> Track {disappeared_id} | "
                                         f"Confidence: {confidence:.0f}% | "
                                         f"Evidence: Proximity={evidence['proximity_score']}, "
                                         f"Evasive={evidence['evasive_score']}, "
                                         f"Impact={evidence['impact_score']}, "
                                         f"Motion={evidence['motion_score']}")

                            # ADAPTIVE threshold: Untuk extreme evasive dengan disappeared victim, threshold lebih rendah
                            angular_vel_aggressor = evasive_data.get('angular_velocity', 0.0)
                            lateral_acc_aggressor = evasive_data.get('lateral_acceleration', 0.0)
                            is_extreme_evasive = (angular_vel_aggressor > 100.0) or (lateral_acc_aggressor > 5.0)
                            
                            # Extreme evasive dengan disappeared victim → threshold lebih rendah (45%)
                            # Normal evasive → threshold normal (50%)
                            max_iou_proximity = proximity_data.get('max_iou', 0.0)
                            adaptive_threshold = 45.0 if (is_extreme_evasive and max_iou_proximity > 0.3) else self.evasive_collision_confidence_threshold
                            
                            if confidence >= adaptive_threshold:
                                # Generate evasive collision detection
                                self._generate_evasive_collision(
                                    aggressor_id=track_id,
                                    victim_id=disappeared_id,
                                    current_frame=current_frame,
                                    evasive_data=evasive_data,
                                    proximity_data=proximity_data,
                                    victim_disappeared=True,
                                    victim_data=disappeared_data,
                                    tracks=tracks,
                                    velocity_field=velocity_field,
                                    detections=detections,
                                    confidence=confidence,
                                    evidence=evidence
                                )

                                # Remove from recently_disappeared (sudah diproses)
                                del self.recently_disappeared[disappeared_id]

                # ============================================
                # Check 2: Active tracks in proximity (DIRECT CHECK - tidak pakai proximity_monitoring)
                # Cek semua track lain yang dekat dengan track yang evasive
                # ============================================
                logger.debug(f"[EVASIVE] Check active tracks in proximity for aggressor {track_id}")

                for other_track in tracks:
                    # Accept both active AND recently-ghosted tracks as potential victims
                    is_valid = self._is_valid_track(other_track)
                    is_ghost_victim = (hasattr(other_track, 'state') and other_track.state == 'ghost' and
                                      hasattr(other_track, 'misses') and other_track.misses <= 3 and
                                      hasattr(other_track, 'hits') and other_track.hits >= 3)
                    if not is_valid and not is_ghost_victim:
                        continue

                    other_id = other_track.track_id
                    if other_id == track_id:
                        continue  # Skip self

                    # Calculate IoU between aggressor and potential victim
                    bbox_aggressor = track.current_detection.get('bbox', [0, 0, 0, 0])
                    bbox_other = other_track.current_detection.get('bbox', [0, 0, 0, 0])
                    iou = self._compute_iou(bbox_aggressor, bbox_other)

                    # Check if close enough (IoU > threshold = MUST OVERLAP significantly)
                    # ADAPTIVE threshold: RELAX for extreme evasive OR massive deceleration
                    angular_vel_aggressor = evasive_data.get('angular_velocity', 0.0)
                    sudden_dec_aggressor = evasive_data.get('sudden_deceleration_percentage', 0.0)
                    is_extreme = angular_vel_aggressor > 100.0
                    has_massive_decel = sudden_dec_aggressor > 80.0

                    # Extreme evasive OR massive deceleration → relax IoU threshold (0.3)
                    # Normal → strict (0.5)
                    iou_threshold = 0.3 if (is_extreme or has_massive_decel) else self.evasive_collision_iou_threshold

                    if iou < iou_threshold:
                        # Fallback: untuk massive decel, cek center-distance proximity
                        # (handles overhead T-bone di persimpangan dimana IoU = 0)
                        if has_massive_decel:
                            _ca = track.current_detection.get('center') or \
                                  [(_bbox_a[0]+_bbox_a[2])/2 for _bbox_a in [bbox_aggressor]][0:1] + \
                                  [(_bbox_a[1]+_bbox_a[3])/2 for _bbox_a in [bbox_aggressor]][0:1]
                            _co = other_track.current_detection.get('center') or \
                                  [(_bbox_o[0]+_bbox_o[2])/2 for _bbox_o in [bbox_other]][0:1] + \
                                  [(_bbox_o[1]+_bbox_o[3])/2 for _bbox_o in [bbox_other]][0:1]
                            _da = ((bbox_aggressor[2]-bbox_aggressor[0])**2 + (bbox_aggressor[3]-bbox_aggressor[1])**2)**0.5
                            _do = ((bbox_other[2]-bbox_other[0])**2 + (bbox_other[3]-bbox_other[1])**2)**0.5
                            _max_d = max(_da, _do, 1.0)
                            _dist = ((_ca[0]-_co[0])**2 + (_ca[1]-_co[1])**2)**0.5
                            if _dist >= _max_d * 2.0:
                                continue
                        else:
                            continue

                    # Track duplication guard: IoU > 0.9 = ByteTrack assigned two IDs
                    # to the same physical object.  Direction change 180° is an artefact
                    # of the duplicate bbox jitter, not a real evasive maneuver.
                    if iou > 0.9:
                        logger.debug(
                            f"[EVASIVE SKIP] Track {track_id} <-> Track {other_id} | "
                            f"IoU={iou:.3f} > 0.9 — track duplication, bukan dua objek berbeda"
                        )
                        continue

                    other_class = other_track.current_detection.get('class_name', 'unknown').lower()

                    # Skip person EXCEPT person+motorcycle
                    aggressor_cls = track.current_detection.get('class_name', 'unknown').lower()
                    if self._is_pedestrian_class(other_class) and not self._is_person_motorcycle_pair(other_class, aggressor_cls):
                        logger.debug(f"[EVASIVE SKIP] Track {other_id} ({other_class}) is person + aggressor ({aggressor_cls}) not motorcycle - skipping")
                        continue

                    logger.warning(f"🔍 EVASIVE PROXIMITY CHECK | "
                                  f"Track {track_id} (evasive) <-> Track {other_id} ({other_class}) | "
                                  f"IoU: {iou:.3f}")

                    # Create temporary proximity data
                    proximity_data = {
                        'max_iou': iou,
                        'frames': [current_frame],
                        'duration': 1
                    }

                    # Calculate confidence (victim still visible)
                    confidence, evidence = self._calculate_collision_confidence(
                        aggressor_track=track,
                        victim_track_id=other_id,
                        victim_track=other_track,
                        proximity_data=proximity_data,
                        evasive_data=evasive_data,
                        victim_disappeared=False,
                        velocity_field=velocity_field,
                        current_frame=current_frame
                    )

                    logger.warning(f"🔍 EVASIVE CONFIDENCE | "
                                  f"Track {track_id} <-> Track {other_id} | "
                                  f"Confidence: {confidence:.0f}% | "
                                  f"Evidence: Proximity={evidence['proximity_score']}, "
                                  f"Evasive={evidence['evasive_score']}, "
                                  f"Impact={evidence['impact_score']}, "
                                  f"Motion={evidence['motion_score']}")

                    # Guard: evasive_score=0 AND impact_score=0 → bukan collision
                    # Pattern FP: Proximity=30, Evasive=0, Impact=0, Motion=15 = total 45%
                    # Ini adalah dua kendaraan mendekat di traffic biasa (closing_rate > 5),
                    # bukan evasive maneuver. Tabrakan nyata selalu punya salah satu:
                    # evasive maneuver (angular vel tinggi / lateral acc tinggi) ATAU
                    # victim impact (velocity/direction change pada korban).
                    _evasive_kinetic_ok = (
                        evidence.get('evasive_score', 0) > 0 or
                        evidence.get('impact_score', 0) > 0
                    )
                    if not _evasive_kinetic_ok:
                        logger.warning(
                            f"⏭️ SKIP EVASIVE COLLISION (NO KINETIC EVIDENCE) | "
                            f"Track {track_id} <-> Track {other_id} | "
                            f"Evasive={evidence.get('evasive_score',0)}, "
                            f"Impact={evidence.get('impact_score',0)} — "
                            f"hanya proximity+motion, bukan collision"
                        )
                        continue

                    # ADAPTIVE threshold: Lower for extreme evasive OR massive deceleration
                    angular_vel_aggressor = evasive_data.get('angular_velocity', 0.0)
                    lateral_acc_aggressor = evasive_data.get('lateral_acceleration', 0.0)
                    sudden_dec_for_threshold = evasive_data.get('sudden_deceleration_percentage', 0.0)
                    is_extreme_evasive = (angular_vel_aggressor > 100.0) or (lateral_acc_aggressor > 5.0)
                    has_massive_decel_for_threshold = sudden_dec_for_threshold > 80.0

                    # Extreme evasive/massive decel with proximity → threshold lebih rendah (45%)
                    # Normal evasive → threshold normal (50%)
                    adaptive_threshold = 45.0 if ((is_extreme_evasive or has_massive_decel_for_threshold) and iou > 0.3) else self.evasive_collision_confidence_threshold

                    if confidence >= adaptive_threshold:
                        # VICTIM VALIDATION DISABLED - rollback per user request
                        aggressor_class = track.current_detection.get('class_name', 'unknown').lower()

                        # FILTER: Skip motorcycle-person stationary pairs (person riding motorcycle)
                        is_motorcycle_person = (
                            ('motorcycle' in aggressor_class and 'person' in other_class) or
                            ('person' in aggressor_class and 'motorcycle' in other_class)
                        )

                        if is_motorcycle_person:
                            # Get velocities
                            aggressor_velocity = velocity_field.compute_velocity(track, dt=1.0)
                            other_velocity = velocity_field.compute_velocity(other_track, dt=1.0)
                            aggressor_speed = np.linalg.norm(aggressor_velocity)
                            other_speed = np.linalg.norm(other_velocity)

                            # Filter if BOTH stationary
                            if aggressor_speed < 0.3 and other_speed < 0.3:
                                logger.warning(f"⏭️ SKIP EVASIVE COLLISION | Motorcycle-person stationary pair | "
                                              f"Track {track_id} (v={aggressor_speed:.2f}) <-> "
                                              f"Track {other_id} (v={other_speed:.2f})")
                                continue

                        # Generate evasive collision detection
                        victim_data = {
                            'last_bbox': other_track.current_detection.get('bbox', [0, 0, 0, 0]),
                            'class_name': other_class
                        }

                        self._generate_evasive_collision(
                            aggressor_id=track_id,
                            victim_id=other_id,
                            current_frame=current_frame,
                            evasive_data=evasive_data,
                            proximity_data=proximity_data,
                            victim_disappeared=False,
                            victim_data=victim_data,
                            tracks=tracks,
                            velocity_field=velocity_field,
                            detections=detections,
                            confidence=confidence,
                            evidence=evidence
                        )

    def _generate_evasive_collision(self, aggressor_id: int, victim_id: int,
                                    current_frame: int, evasive_data: Dict,
                                    proximity_data: Dict, victim_disappeared: bool,
                                    victim_data: Dict, tracks: List,
                                    velocity_field: 'VelocityField', detections: List[Dict],
                                    confidence: float, evidence: Dict) -> None:
        """
        Generate collision detection untuk evasive collision dengan CONFIDENCE SCORING

        Args:
            aggressor_id: ID mobil yang oleng (aggressor)
            victim_id: ID victim (motor/person)
            current_frame: Frame number saat ini
            evasive_data: Evasive maneuver data (angular_velocity, lateral_accel, etc.)
            proximity_data: Data proximity history
            victim_disappeared: True jika victim hilang
            victim_data: Data victim (last_bbox, class_name)
            tracks: List of active tracks
            velocity_field: VelocityField instance
            detections: List untuk append hasil detection
            confidence: Collision confidence score (0-100)
            evidence: Evidence breakdown dict
        """
        # Find aggressor track
        aggressor_track = None
        for track in tracks:
            if track.track_id == aggressor_id:
                aggressor_track = track
                break

        if aggressor_track is None:
            return

        # Get track info
        aggressor_bbox = aggressor_track.current_detection.get('bbox', [0, 0, 0, 0])
        victim_bbox = victim_data['last_bbox']
        aggressor_class = aggressor_track.current_detection.get('class_name', 'unknown')
        victim_class = victim_data['class_name']

        # Calculate collision point (use last known position of victim)
        collision_point = self._compute_collision_point(aggressor_bbox, victim_bbox)

        # Get velocities
        v_aggressor = velocity_field.compute_velocity(aggressor_track, dt=1.0)
        v_aggressor_speed = np.linalg.norm(v_aggressor)

        # Create detection dengan CONFIDENCE SCORING
        victim_status = 'disappeared' if victim_disappeared else 'visible'

        # Energy loss based on victim status
        if victim_disappeared:
            energy_loss_secondary = 1.0  # 100% (disappeared)
        else:
            # Estimate from victim velocity change
            victim_velocity_change = evidence.get('victim_velocity_change', 0.0)
            energy_loss_secondary = min(victim_velocity_change, 1.0)

        evasive_escalated = False
        evasive_escalation_rule = ''
        evasive_direction_change = float(evasive_data.get('direction_change', 0.0))
        evasive_iou = float(proximity_data.get('max_iou', 0.0))
        evasive_max_kelr = max(0.3, float(energy_loss_secondary))
        if (
            self.evasive_escalation_enabled
            and confidence < self.collision_confidence_threshold
            and evasive_direction_change >= self.evasive_escalation_direction_threshold
            and evasive_iou >= self.evasive_escalation_iou_threshold
            and evasive_max_kelr <= self.evasive_escalation_kelr_max
        ):
            confidence = max(confidence, self.collision_confidence_threshold)
            evasive_escalated = True
            evasive_escalation_rule = (
                f"direction_change>={self.evasive_escalation_direction_threshold} "
                f"AND IoU>={self.evasive_escalation_iou_threshold} "
                f"AND KELR<={self.evasive_escalation_kelr_max}"
            )
            logger.warning(
                f"[EVASIVE ESCALATION] Frame {current_frame}: "
                f"Track {aggressor_id} <-> Track {victim_id} | "
                f"Rule: {evasive_escalation_rule} | "
                f"direction_change={evasive_direction_change:.1f} | "
                f"IoU={evasive_iou:.4f} | "
                f"max_kelr={evasive_max_kelr:.3f} | "
                f"Orange critical interaction upgraded to red escalation"
            )

        low_fragment_upgraded = False
        low_fragment_tracks_created = int(getattr(self, 'total_tracks_created_so_far', 0) or 0)
        low_fragment_active_tracks = len(tracks) if tracks is not None else 0
        low_fragment_kelr_max = max(0.3, float(energy_loss_secondary))
        if (
            self.low_fragment_escalation_enabled
            and confidence >= 50
            and confidence < self.collision_confidence_threshold
            and low_fragment_tracks_created <= self.low_fragment_track_threshold
            and low_fragment_active_tracks <= self.low_fragment_active_track_threshold
            and low_fragment_kelr_max >= self.low_fragment_kelr_min
        ):
            confidence = max(confidence, self.collision_confidence_threshold)
            low_fragment_upgraded = True
            logger.warning(
                f"[LOW-FRAGMENT ESCALATION] Frame {current_frame}: "
                f"Track {aggressor_id} <-> Track {victim_id} | "
                f"tracks_created={low_fragment_tracks_created} <= {self.low_fragment_track_threshold} | "
                f"active_tracks={low_fragment_active_tracks} <= {self.low_fragment_active_track_threshold} | "
                f"kelr_max={low_fragment_kelr_max:.3f} >= {self.low_fragment_kelr_min} | "
                f"Critical interaction upgraded to red escalation"
            )

        confidence, scene_guard_promoted, scene_guard_suppressed = self._apply_global_red_scene_guard(
            confidence=confidence,
            tracks=tracks,
            frame_id=current_frame,
            pair_desc=f"Track {aggressor_id} <-> Track {victim_id}",
            context='evasive_collision',
        )

        # Severity based on confidence (configurable threshold)
        if confidence >= self.collision_confidence_threshold:
            severity = 'critical'  # High confidence
            alert_level = 'emergency'
            confidence_label = 'COLLISION DETECTED'
        elif confidence >= 50:
            severity = 'high'      # Medium confidence
            alert_level = 'warning'
            confidence_label = 'CRITICAL INTERACTION'
        else:
            severity = 'medium'    # Low confidence (shouldn't reach here)
            alert_level = 'caution'
            confidence_label = 'UNSAFE PROXIMITY'

        # Union bbox untuk visualisasi
        _union_bbox = [
            min(aggressor_bbox[0], victim_bbox[0]),
            min(aggressor_bbox[1], victim_bbox[1]),
            max(aggressor_bbox[2], victim_bbox[2]),
            max(aggressor_bbox[3], victim_bbox[3])
        ]

        detection = {
            'track_id': aggressor_id,
            'track_id_secondary': victim_id,
            'behaviour_type': 'collision',
            'severity': severity,
            'collision_point': collision_point,
            'detection_mode': 'evasive_collision',
            'bbox': _union_bbox,
            'prediction_mode': 'confirmed',
            'alert_level': alert_level,
            'confidence_label': confidence_label,
            'class_primary': aggressor_class,
            'class_secondary': victim_class,

            # CONFIDENCE SCORING (NEW!)
            'confidence': confidence,
            'confidence_level': 'HIGH' if confidence >= self.collision_confidence_threshold else 'MEDIUM',
            'evidence_breakdown': evidence,
            'evasive_escalated': evasive_escalated,
            'evasive_escalation_rule': evasive_escalation_rule,
            'low_fragment_escalated': low_fragment_upgraded,
            'low_fragment_tracks_created': low_fragment_tracks_created,
            'evasive_escalation_direction_change': evasive_direction_change,
            'evasive_escalation_iou': evasive_iou,
            'evasive_escalation_max_kelr': evasive_max_kelr,

            # Evasive maneuver data
            'evasive_data': evasive_data,

            # Proximity history
            'proximity_frames': len(proximity_data.get('frames', [])),
            'max_iou': proximity_data.get('max_iou', 0.0),

            # Energy loss
            'energy_loss_primary': 0.3,    # Aggressor (approximate)
            'energy_loss_secondary': energy_loss_secondary,

            # Other data
            'iou_overlap': proximity_data.get('max_iou', 0.0),
            'frame_id': current_frame,
            'class_i': aggressor_class,
            'class_j': victim_class,

            # Victim status
            'victim_status': victim_status,
            'victim_disappeared': victim_disappeared,

            # Description
            'description': (f"Evasive collision ({confidence:.0f}% confidence): "
                          f"{aggressor_class} (ID:{aggressor_id}) swerved suddenly "
                          f"(ω={evasive_data['angular_velocity']:.1f}°/fr, "
                          f"a_lat={evasive_data['lateral_acceleration']:.2f} px/fr²) and "
                          f"{victim_class} (ID:{victim_id}) {victim_status}")
        }

        # Per-frame pair deduplication: sorted(A,B) hanya boleh confirm sekali per frame.
        # Ini eliminasi (A→B) + (B→A) double-report dan mencegah spam saat banyak kendaraan
        # di scene yang sama saling flag satu sama lain.
        pair_key = tuple(sorted([aggressor_id, victim_id]))
        if not hasattr(self, '_evasive_frame_pairs'):
            self._evasive_frame_pairs = {}
        if self._evasive_frame_pairs.get(pair_key) == current_frame:
            logger.debug(f"[EVASIVE DEDUP] Pair ({aggressor_id},{victim_id}) already confirmed frame {current_frame} — skip")
            return
        self._evasive_frame_pairs[pair_key] = current_frame

        # Cross-frame cooldown: pair yang sudah dikonfirmasi tidak boleh api lagi
        # dalam 30 frame berikutnya (≈1 detik). Mencegah spam repeated confirmation
        # untuk kejadian yang sama (satu collision = satu alert, bukan per-frame alert).
        if not hasattr(self, '_evasive_pair_cooldown'):
            self._evasive_pair_cooldown = {}
        _last_confirmed = self._evasive_pair_cooldown.get(pair_key, -999)
        _COOLDOWN_FRAMES = 30
        if current_frame - _last_confirmed < _COOLDOWN_FRAMES:
            logger.debug(f"[EVASIVE COOLDOWN] Pair ({aggressor_id},{victim_id}) cooldown "
                         f"({current_frame - _last_confirmed}/{_COOLDOWN_FRAMES} frames) — skip")
            return
        self._evasive_pair_cooldown[pair_key] = current_frame

        # Aggressor-level cooldown: satu aggressor hanya boleh confirm SATU victim per window.
        # Mencegah satu kendaraan generate N events sekaligus terhadap N kendaraan di sekitarnya.
        # Fisika: satu kendaraan hanya bisa terlibat satu collision event dalam satu waktu.
        if not hasattr(self, '_evasive_aggressor_cooldown'):
            self._evasive_aggressor_cooldown = {}
        _last_agg = self._evasive_aggressor_cooldown.get(aggressor_id, -999)
        if current_frame - _last_agg < _COOLDOWN_FRAMES:
            logger.debug(f"[EVASIVE AGGRESSOR COOLDOWN] Track {aggressor_id} cooldown "
                         f"({current_frame - _last_agg}/{_COOLDOWN_FRAMES} frames) — skip")
            return
        self._evasive_aggressor_cooldown[aggressor_id] = current_frame

        # Global evasive cooldown: setelah SATU event confirmed, blokir semua evasive baru
        # selama N frame. Mencegah flood dari banyak aggressor baru yang muncul bersamaan
        # (e.g., semua kendaraan di intersection berhenti serentak saat ada kecelakaan).
        # Filosofi: satu collision event = satu alert window. Detector tidak perlu report
        # setiap kendaraan yang terkena dampak — cukup satu sinyal per kejadian.
        if not hasattr(self, '_evasive_global_cooldown'):
            self._evasive_global_cooldown = -999
        _GLOBAL_COOLDOWN = 90  # 2 detik pada 30fps, 1 detik pada 60fps
        if current_frame - self._evasive_global_cooldown < _GLOBAL_COOLDOWN:
            logger.debug(f"[EVASIVE GLOBAL COOLDOWN] Frame {current_frame} | "
                         f"last={self._evasive_global_cooldown} | "
                         f"remaining={_GLOBAL_COOLDOWN - (current_frame - self._evasive_global_cooldown)} frames")
            return
        self._evasive_global_cooldown = current_frame

        detections.append(detection)

        # Increment detection counter
        self.total_detections += 1

        # Log collision dengan confidence
        logger.warning(f"[EVASIVE COLLISION CONFIRMED] Frame {current_frame}: "
                      f"Track {aggressor_id} ({aggressor_class}) <-> Track {victim_id} ({victim_class}) | "
                      f"Confidence: {confidence:.0f}% ({severity.upper()}) | "
                      f"Angular velocity: {evasive_data['angular_velocity']:.1f}°/fr | "
                      f"Lateral accel: {evasive_data['lateral_acceleration']:.2f} px/fr² | "
                      f"Direction change: {evasive_data['direction_change']:.1f}° | "
                      f"Victim status: {victim_status} | "
                      f"Evidence: Proximity={evidence['proximity_score']}, "
                      f"Evasive={evidence['evasive_score']}, "
                      f"Impact={evidence['impact_score']}, "
                      f"Motion={evidence['motion_score']}")
        if pair_key in self.proximity_monitoring:
            del self.proximity_monitoring[pair_key]

    def _detect_stationary_behavior(self, track: object, velocity_field: 'VelocityField',
                                    frames_to_check: int = 3) -> Tuple[bool, float, int]:
        """
        Detect if track has been stationary (velocity ≈ 0) for N consecutive frames

        This is POST-IMPACT behavior indicator:
        - Motor jatuh setelah tabrakan → velocity drop to ~0 and stays stationary
        - Mobil berhenti setelah tabrakan → velocity drop to ~0 and stays stationary

        Args:
            track: Track object to check
            velocity_field: VelocityField instance
            frames_to_check: Number of consecutive frames to check (default: 3)

        Returns:
            Tuple[bool, float, int]:
                - is_stationary: True if track has been stationary for N frames
                - avg_velocity: Average velocity magnitude over the checked frames
                - stationary_duration: Number of consecutive stationary frames
        """
        STATIONARY_THRESHOLD = 0.5  # px/fr - velocity below this = stationary

        # Need at least frames_to_check history
        if len(track.history) < frames_to_check:
            return False, 0.0, 0

        # Check velocity for last N frames
        velocities = []
        for i in range(frames_to_check):
            if i == 0:
                # Current frame
                v = velocity_field.compute_velocity(track, dt=1.0)
            else:
                # Past frames
                original = track.current_detection
                track.current_detection = track.history[-i]
                v = velocity_field.compute_velocity(track, dt=1.0)
                track.current_detection = original

            speed = np.linalg.norm(v)
            velocities.append(speed)

        # Check if ALL velocities are below threshold (stationary)
        avg_velocity = np.mean(velocities)
        is_stationary = all(v < STATIONARY_THRESHOLD for v in velocities)
        stationary_duration = frames_to_check if is_stationary else 0

        return is_stationary, avg_velocity, stationary_duration

    def _check_post_impact_collision(self, tracks: List, velocity_field: 'VelocityField',
                                     current_frame: int, detections: List[Dict]) -> None:
        """
        Check untuk POST-IMPACT collision detection:
        - Track yang stationary (velocity ≈ 0 for 3+ frames)
        - PLUS ada track lain di proximity (IoU > 0.2)
        - PLUS ada evidence of prior impact (rotation spike, trajectory change, etc.)

        Ini handle kasus:
        - Motor jatuh setelah ditabrak, tapi MASIH terdeteksi YOLO (frame 95 problem!)
        - Motor velocity ≈ 0, tapi masih visible di frame

        Physics Evidence:
        1. Stationary behavior (velocity ≈ 0 for 3+ frames)
        2. Proximity with another track (IoU > 0.2 OR distance < threshold)
        3. Prior impact evidence:
           - High rotation spike in recent history (sudden spin)
           - Large trajectory change (direction change > 30°)
           - Sudden deceleration in recent history (velocity dropped > 70%)

        Args:
            tracks: List of active tracks
            velocity_field: VelocityField instance
            current_frame: Current frame number
            detections: List to append collision detection results
        """
        # Loop through all tracks to find stationary tracks
        for track_i in tracks:
            if not self._is_valid_track(track_i):
                continue

            # Check if track is stationary
            is_stationary, avg_velocity, stationary_duration = self._detect_stationary_behavior(
                track_i, velocity_field, frames_to_check=3
            )

            if not is_stationary:
                continue  # Not stationary, skip

            track_i_id = track_i.track_id
            track_i_class = track_i.current_detection.get('class_name', 'unknown')
            bbox_i = track_i.current_detection.get('bbox', [0, 0, 0, 0])

            # Track is stationary - check for proximity with other tracks
            for track_j in tracks:
                if not self._is_valid_track(track_j):
                    continue

                track_j_id = track_j.track_id
                if track_i_id == track_j_id:
                    continue  # Skip self

                # Calculate IoU
                bbox_j = track_j.current_detection.get('bbox', [0, 0, 0, 0])
                iou = self._compute_iou(bbox_i, bbox_j)

                # Check proximity (IoU > 0.2 = close enough)
                if iou < 0.2:
                    continue  # Not close enough

                track_j_class = track_j.current_detection.get('class_name', 'unknown')

                # Skip person pairs EXCEPT person+motorcycle
                has_person = self._is_pedestrian_class(track_i_class.lower()) or self._is_pedestrian_class(track_j_class.lower())
                if has_person and not self._is_person_motorcycle_pair(track_i_class.lower(), track_j_class.lower()):
                    continue

                # Check for prior impact evidence (rotation spike, trajectory change, sudden deceleration)
                has_impact_evidence = False
                impact_evidence_details = []

                # Evidence 1: Rotation spike (sudden spin)
                rotation_spike_i = self._compute_rotation_spike(track_i, velocity_field)
                rotation_spike_j = self._compute_rotation_spike(track_j, velocity_field)
                if rotation_spike_i > 0.5 or rotation_spike_j > 0.5:
                    has_impact_evidence = True
                    impact_evidence_details.append(f"Rotation spike: {max(rotation_spike_i, rotation_spike_j):.2f}")

                # Evidence 2: Sudden deceleration (check velocity history)
                # For stationary track, check if it was moving before (sudden stop = impact)
                if track_i_id in self.track_motion_history:
                    history = self.track_motion_history[track_i_id]
                    if len(history['velocities']) >= 3:
                        velocities = list(history['velocities'])
                        # Check velocity 3 frames ago vs current
                        speed_previous = np.linalg.norm(velocities[-3]) if len(velocities) >= 3 else 0.0
                        speed_current = avg_velocity

                        if speed_previous > 1.0:  # Was moving
                            velocity_drop = speed_previous - speed_current
                            velocity_drop_pct = (velocity_drop / speed_previous) if speed_previous > 0 else 0.0

                            if velocity_drop_pct > 0.7:  # Dropped > 70%
                                has_impact_evidence = True
                                impact_evidence_details.append(
                                    f"Sudden deceleration: {speed_previous:.2f} → {speed_current:.2f} "
                                    f"({velocity_drop_pct*100:.0f}% drop)"
                                )

                # Evidence 3: Check if track_j also has evasive maneuver or sudden behavior
                if track_j_id in self.track_motion_history:
                    v_j = velocity_field.compute_velocity(track_j, dt=1.0)
                    is_evasive_j, angular_vel_j, lateral_acc_j, dir_change_j, sudden_dec_pct_j = self._detect_evasive_maneuver(
                        track_j_id, v_j
                    )
                    if is_evasive_j:
                        has_impact_evidence = True
                        impact_evidence_details.append(
                            f"Track {track_j_id} evasive: ω={angular_vel_j:.1f}°/fr, "
                            f"a_lat={lateral_acc_j:.2f} px/fr²"
                        )

                # Only generate collision if we have impact evidence
                if not has_impact_evidence:
                    continue

                # Generate collision detection
                logger.warning(f"🔴 POST-IMPACT COLLISION DETECTED | Frame {current_frame} | "
                              f"Track {track_i_id} ({track_i_class}) STATIONARY (v={avg_velocity:.2f} px/fr) | "
                              f"In proximity with Track {track_j_id} ({track_j_class}) | "
                              f"IoU: {iou:.3f} | Evidence: {', '.join(impact_evidence_details)}")

                # Calculate collision point
                collision_point = self._compute_collision_point(bbox_i, bbox_j)

                # Get velocities
                v_i = velocity_field.compute_velocity(track_i, dt=1.0)
                v_j = velocity_field.compute_velocity(track_j, dt=1.0)
                v_i_speed = np.linalg.norm(v_i)
                v_j_speed = np.linalg.norm(v_j)

                # Create detection
                detection = {
                    'behaviour': BehaviourState.CONFIRMED,
                    'track_id': track_i_id,
                    'track_id_secondary': track_j_id,
                    'severity': 'high',  # High severity for post-impact (stationary victim)
                    'collision_point': collision_point,
                    'detection_mode': 'post_impact_stationary',

                    # Physics data
                    'iou_overlap': iou,
                    'energy_loss_primary': 1.0,    # Stationary = 100% energy loss
                    'energy_loss_secondary': 0.3,  # Approximate for other track
                    'frame_id': current_frame,
                    'class_i': track_i_class,
                    'class_j': track_j_class,

                    # Post-impact specific data
                    'stationary_duration': stationary_duration,
                    'avg_velocity': avg_velocity,
                    'impact_evidence': impact_evidence_details,
                    'rotation_spike_i': rotation_spike_i,
                    'rotation_spike_j': rotation_spike_j,

                    # Velocities
                    'velocity_i': v_i_speed,
                    'velocity_j': v_j_speed,

                    'description': (f"Post-impact collision: {track_i_class} (ID:{track_i_id}) "
                                  f"stationary for {stationary_duration} frames (v={avg_velocity:.2f} px/fr), "
                                  f"in proximity with {track_j_class} (ID:{track_j_id}). "
                                  f"Evidence: {', '.join(impact_evidence_details)}")
                }

                detections.append(detection)
                self.total_detections += 1

                # Only report once per pair
                break


# Testing
if __name__ == "__main__":
    print("Testing CollisionDetector...")
    
    # Create detector
    config = {
        'iou_overlap_threshold': 0.3,
        'energy_loss_threshold': 0.9,
        'variance_acceleration': 5.0,
        'window_size': 3,
        'persist_threshold': 0.8
    }
    
    detector = CollisionDetector(config)
    
    # Create dummy tracks - two cars colliding
    class DummyTrack:
        def __init__(self, track_id, centers, bboxes):
            self.track_id = track_id
            self.state = 'active'
            self.hits = 10
            self.current_frame = len(centers)
            
            self.history = [{'center': centers[i], 'bbox': bboxes[i]} 
                           for i in range(len(centers)-1)]
            self.current_detection = {
                'center': centers[-1],
                'bbox': bboxes[-1],
                'class_name': 'car'
            }
    
    # Car 1: Moving right, then stops (collision)
    car1_centers = [[100+i*10, 100] for i in range(5)] + [[150, 100], [152, 100]]
    car1_bboxes = [[c[0]-20, c[1]-15, c[0]+20, c[1]+15] for c in car1_centers]
    car1 = DummyTrack(1, car1_centers, car1_bboxes)
    
    # Car 2: Moving left, then stops (collision) - overlapping with car1
    car2_centers = [[200-i*10, 105] for i in range(5)] + [[150, 105], [152, 105]]
    car2_bboxes = [[c[0]-20, c[1]-15, c[0]+20, c[1]+15] for c in car2_centers]
    car2 = DummyTrack(2, car2_centers, car2_bboxes)
    
    # Create velocity field
    from physics.velocity_field import VelocityField
    vf_config = {'gaussian_radius': 100.0, 'flow_sigma': 50.0}
    vf = VelocityField(vf_config)
    
    print("\nTesting collision detection:")
    
    # Check IoU overlap
    iou = detector._compute_iou(car1.current_detection['bbox'], 
                                car2.current_detection['bbox'])
    print(f"IoU overlap: {iou:.3f} (threshold: {detector.iou_overlap_threshold})")
    
    # Check energy loss
    energy_loss_1 = detector._compute_energy_loss(car1, vf)
    energy_loss_2 = detector._compute_energy_loss(car2, vf)
    print(f"Energy loss car1: {energy_loss_1:.3f}")
    print(f"Energy loss car2: {energy_loss_2:.3f}")
    
    # Full detection
    tracks = [car1, car2]
    detections = detector.detect(tracks, vf)
    
    print(f"\n✓ Total collision detections: {len(detections)}")
    for det in detections:
        print(f"  Collision: Track {det['track_id']} <-> Track {det['track_id_secondary']}")
        print(f"    IoU: {det['iou_overlap']:.3f}")
        print(f"    Energy loss: {det['energy_loss_primary']:.3f}, {det['energy_loss_secondary']:.3f}")
        print(f"    Collision point: {det['collision_point']}")
        print(f"    Severity: {det['severity']}")
    
    # Statistics
    stats = detector.get_statistics()
    print(f"\n✓ CollisionDetector Statistics:")
    print(f"  Confirmed detections: {stats['confirmed_detections']}")
    print(f"  Tracked collision pairs: {len(detector.collision_pairs)}")
    
    print("\n✓ CollisionDetector test completed")

