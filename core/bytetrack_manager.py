"""
ByteTrack Manager - BYTETracker Integration for TrackGuard
==========================================================

Wrapper yang mengintegrasikan BYTETracker ke pipeline TrackGuard.
Menggantikan Smart Hungarian tracker dengan ByteTrack untuk tracking
yang lebih stabil (mengurangi ID switch dan duplicate detection).

PIPELINE:
YOLOv8 → ByteTrack Association → Track Conversion → Physics/Behaviour Detection
"""

import sys
import os
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import logging
from pathlib import Path

# Add bytetrack to path for imports. Use a resolved absolute path because SSH
# batch runs can otherwise miss the nested yolox package.
_BYTETRACK_ROOT = Path(__file__).resolve().parents[1] / 'bytetrack'
if str(_BYTETRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BYTETRACK_ROOT))

from core.detector import YOLOv8Detector
from core.track_manager import Track, KalmanFilter
from types import SimpleNamespace

logger = logging.getLogger(__name__)


def _compute_iou(bbox_a, bbox_b):
    """Compute IoU between two bboxes [x1,y1,x2,y2]"""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class ByteTrackManager:
    """
    ByteTrack-based TrackManager yang compatible dengan TrackGuard physics pipeline.

    Drop-in replacement untuk PureSmartHungarianTrackManager — expose interface
    yang sama (process_frame, get_current_tracks, behaviour_results, dll).
    """

    def __init__(self, config=None, use_physics=False):
        from utils.settings import SETTINGS

        # === BYTETRACK INITIALIZATION ===
        bt_args = SimpleNamespace(
            track_thresh=0.5,    # High confidence threshold
            track_buffer=30,     # Frames to keep lost tracks
            match_thresh=0.8,    # IoU matching threshold
            mot20=False
        )

        from yolox.tracker.byte_tracker import BYTETracker, BaseTrack
        BaseTrack._count = 0  # Reset track ID counter
        self.byte_tracker = BYTETracker(bt_args, frame_rate=30)
        print("✓ BYTETracker initialized (track_thresh=0.5, buffer=30, match=0.8)")

        # === DETECTOR (same as PureSmartHungarianTrackManager) ===
        self.detector = YOLOv8Detector()
        print("✓ YOLOv8 Detection Pipeline initialized")

        # === TRACK STORAGE ===
        self.tracks = {}           # Dict[track_id, Track]
        self.confirmed_tracks = {} # Alias for compatibility

        # === PHYSICS INTEGRATION (copy from PureSmartHungarianTrackManager) ===
        self.use_physics = use_physics
        self.physics_config = SETTINGS.PHYSICS_CONFIG if use_physics else None
        self.behaviour_detectors = None
        self.physics_predictor = None
        self.eager_smoother = None
        self.velocity_field = None
        self.scene_analyzer = None

        # R2Q4 split profiling state (per frame).
        self._frame_ppl_compute_time = 0.0
        self._frame_alarm_decision_time = 0.0
        self._frame_collision_detector_time = 0.0

        if use_physics:
            print("🔬 LTE-TrackGuard Physics Mode ENABLED (ByteTrack)")
            print(f"   EAGER smoothing: {self.physics_config['enable_eager']}")
            print(f"   Physics predictor: {self.physics_config['enable_physics_predictor']}")
            print(f"   Behaviour detection: {self.physics_config['enable_behaviour_detection']}")

            from physics.velocity_field import VelocityField
            self.velocity_field = VelocityField(self.physics_config['velocity_field'])
            print("   ✓ VelocityField loaded")

            from physics.scene_analyzer import SceneAnalyzer
            self.scene_analyzer = SceneAnalyzer(self.physics_config['scene_analyzer'])
            print("   ✓ SceneAnalyzer loaded")

            if self.physics_config['enable_eager']:
                from physics.eager import EAGERSmoother
                self.eager_smoother = EAGERSmoother(self.physics_config['eager'])
                print("   ✓ EAGER Smoother loaded")

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

            self.behaviour_results = {
                'turn': [], 'brake': [], 'fallen': [], 'collision': []
            }
        else:
            print("📊 Standard ByteTrack Mode (no physics)")
            self.behaviour_results = {}

        # === STATISTICS ===
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_associations = 0
        self.total_id_switches = 0
        self.total_ghost_reidentifications = 0
        self.pipeline_stats = {
            'detection_time': [],
            'tracking_time': [],
            'physics_time': [],
            'total_pipeline_time': []
        }

        print("🎯 ByteTrack Manager initialized")
        print("   Pipeline: YOLOv8 → BYTETracker → Track Conversion → Physics")

    def process_frame(self, image: np.ndarray, frame_id: int) -> Dict:
        """Main pipeline — compatible dengan PureSmartHungarianTrackManager"""
        self.frame_count += 1
        pipeline_start = time.time()

        # R2Q4 per-frame split timing populated by CollisionDetector.
        self._frame_ppl_compute_time = 0.0
        self._frame_alarm_decision_time = 0.0
        self._frame_collision_detector_time = 0.0

        # === 1. DETECTION ===
        det_start = time.time()
        detections = self.detector.detect(image)
        det_time = time.time() - det_start

        # === 2. BYTETRACK UPDATE ===
        track_start = time.time()

        if len(detections) > 0:
            # Convert detections → ByteTrack format [x1, y1, x2, y2, score]
            output_results = np.array([
                [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['confidence']]
                for d in detections
            ], dtype=np.float32)
        else:
            output_results = np.empty((0, 5), dtype=np.float32)

        img_h, img_w = image.shape[:2]
        # img_info and img_size same (no scaling needed, YOLO already handles it)
        online_stracks = self.byte_tracker.update(
            output_results,
            img_info=[img_h, img_w],
            img_size=[img_h, img_w]
        )

        track_time = time.time() - track_start

        # === 3. CONVERT STrack → Track ===
        current_track_ids = set()

        for strack in online_stracks:
            tid = strack.track_id
            current_track_ids.add(tid)

            # Match STrack bbox to original detection for class_name
            strack_bbox = strack.tlbr  # [x1, y1, x2, y2]
            matched_det = self._match_strack_to_detection(strack_bbox, detections)

            if matched_det is None:
                # Fallback: construct detection from STrack data
                matched_det = {
                    'bbox': [int(strack_bbox[0]), int(strack_bbox[1]),
                             int(strack_bbox[2]), int(strack_bbox[3])],
                    'confidence': float(strack.score),
                    'center': [(strack_bbox[0] + strack_bbox[2]) / 2,
                               (strack_bbox[1] + strack_bbox[3]) / 2],
                    'size': [strack_bbox[2] - strack_bbox[0],
                             strack_bbox[3] - strack_bbox[1]],
                    'aspect_ratio': (strack_bbox[3] - strack_bbox[1]) / max(strack_bbox[2] - strack_bbox[0], 1),
                    'class_name': 'unknown',
                    'class_id': -1
                }
            else:
                # Use RAW YOLO detection bbox for physics computation
                # ByteTrack Kalman filter smooths positions which dampens collision signals
                # (IoU overlap, velocity drop, energy loss) — so we keep raw YOLO bbox
                # ByteTrack is used ONLY for ID association, not position filtering
                matched_det = matched_det.copy()

            if tid in self.tracks:
                # Update existing track
                self.tracks[tid].update(matched_det, np.array([]), frame_id)
                self.total_associations += 1
            else:
                # Create new track
                track = Track(tid, matched_det, np.array([]), frame_id)

                # ── History inheritance ──────────────────────────────────────
                # Problem: ByteTrack loses a track at collision impact (rapid
                # bbox deformation → confidence drop → ID lost), then
                # re-detects with a new ID.  New track has 0 history →
                # KELR/ARS have no baseline → PPL blind at exactly the moment
                # it needs to fire.
                #
                # Fix: if a new track appears within 1.5× bbox-diagonal of a
                # recently-ghosted track (misses ≤ 5 frames), it is almost
                # certainly the same physical vehicle re-detected after the
                # impact deformation.  Inherit the ghost's history so PPL
                # retains its kinematic baseline.
                #
                # Guard: only inherit from the single nearest ghost; if
                # multiple ghosts qualify, pick the closest.  Ghost is then
                # terminated so it is not double-counted.
                _new_center = np.array(matched_det['center'], dtype=float)
                _nb = matched_det['bbox']
                _new_diag = math.sqrt(
                    max(_nb[2] - _nb[0], 1) ** 2 +
                    max(_nb[3] - _nb[1], 1) ** 2
                )
                _best_ghost = None
                _best_dist  = float('inf')
                for _gtid, _gt in self.tracks.items():
                    if _gt.state != 'ghost' or _gt.misses > 5:
                        continue
                    if not _gt.history:
                        continue
                    _gc = np.array(_gt.history[-1]['center'], dtype=float)
                    _gb = _gt.history[-1].get('bbox', _nb)
                    _g_diag = math.sqrt(
                        max(_gb[2] - _gb[0], 1) ** 2 +
                        max(_gb[3] - _gb[1], 1) ** 2
                    )
                    _dist = float(np.linalg.norm(_new_center - _gc))
                    if _dist < 1.5 * max(_new_diag, _g_diag) and _dist < _best_dist:
                        _best_dist  = _dist
                        _best_ghost = _gt

                if _best_ghost is not None:
                    track.history = list(_best_ghost.history)
                    _best_ghost.state = 'terminated'   # prevent double-count
                    logger.debug(
                        f"[HISTORY INHERIT] New track {tid} inherited "
                        f"{len(track.history)} frames from ghost "
                        f"(dist={_best_dist:.1f}px)"
                    )
                # ────────────────────────────────────────────────────────────

                self.tracks[tid] = track
                self.total_tracks_created += 1

        # === 3.5 GHOST TRACK EXPOSURE (HYBRID) ===
        # Expose ByteTrack's lost_stracks as ghost tracks for collision detector
        # This provides disappearance signals WITHOUT tracker noise/duplicate IDs
        lost_track_ids = set()
        for lost_strack in self.byte_tracker.lost_stracks:
            lost_tid = lost_strack.track_id
            lost_track_ids.add(lost_tid)

            if lost_tid in self.tracks and self.tracks[lost_tid].state == 'active':
                # Track just went from active → ghost (ByteTrack lost it)
                self.tracks[lost_tid].state = 'ghost'
                self.tracks[lost_tid].misses += 1
                self.tracks[lost_tid].time_since_update += 1

        # Mark remaining tracks not in online or lost as terminated
        for tid in list(self.tracks.keys()):
            if tid not in current_track_ids and tid not in lost_track_ids:
                track = self.tracks[tid]
                if track.state == 'active':
                    track.state = 'ghost'
                    track.misses += 1
                    track.time_since_update += 1
                elif track.state == 'ghost':
                    track.misses += 1
                    track.time_since_update += 1
                    if track.misses > 30:
                        track.state = 'terminated'

        # Clean terminated tracks
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.state != 'terminated'}

        # Update confirmed_tracks (alias)
        self.confirmed_tracks = {tid: t for tid, t in self.tracks.items() if t.state == 'active'}

        # === 4. PHYSICS / BEHAVIOUR DETECTION ===
        # HYBRID: Feed BOTH active + recently-ghosted tracks to collision detector
        # Ghost tracks (just disappeared) provide disappearance/energy-loss signals
        # that the collision detector needs for _check_disappearance_collision
        physics_start = time.time()
        active_tracks = self.get_current_tracks()
        ghost_tracks = [t for t in self.tracks.values()
                       if t.state == 'ghost' and t.misses <= 3]  # Only recently-ghosted (<=3 frames)
        physics_tracks = active_tracks + ghost_tracks

        if self.use_physics and self.velocity_field is not None:
            # VelocityField computes velocity on-the-fly from track history
            # No explicit update needed — compute_velocity() reads track.history directly

            # Scene analysis
            scene_analysis = None
            if self.scene_analyzer is not None:
                frame_height, frame_width = image.shape[:2]
                scene_analysis = self.scene_analyzer.analyze_scene(
                    active_tracks, (frame_height, frame_width)
                )

            # Behaviour detection
            if self.behaviour_detectors is not None:
                # Apply adaptive thresholds
                original_thresholds = None
                if scene_analysis is not None:
                    original_thresholds = self.scene_analyzer.apply_adaptive_thresholds(
                        self.behaviour_detectors,
                        scene_analysis['adaptive_params']
                    )

                # Run detectors (collision first, then others)
                collision_track_ids = set()

                for detector_name, detector in self.behaviour_detectors.items():
                    if detector_name == 'collision':
                        # HYBRID: Feed physics_tracks (active + ghost) to collision detector
                        # Ghost tracks provide disappearance signals for collision detection
                        detector.total_tracks_created_so_far = self.total_tracks_created
                        _collision_profile_start = time.perf_counter()
                        results = detector.detect(physics_tracks, self.velocity_field)
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
                        for det in results:
                            collision_track_ids.add(det.get('track_id', -1))
                            collision_track_ids.add(det.get('track_id_secondary', -1))
                    elif detector_name == 'fallen':
                        brake_results = self.behaviour_results.get('brake', [])
                        turn_results = self.behaviour_results.get('turn', [])

                        if len(brake_results) > 0 or len(turn_results) > 0 or frame_id % 30 == 0:
                            logger.warning(f"[DEBUG] Frame {frame_id}: Brake detections: {len(brake_results)}, "
                                         f"Turn detections: {len(turn_results)}, Active tracks: {len(active_tracks)}")

                        results = detector.detect(
                            active_tracks, self.velocity_field,
                            brake_results=brake_results, turn_results=turn_results
                        )
                        self.behaviour_results[detector_name] = results
                    else:
                        results = detector.detect(active_tracks, self.velocity_field)

                        # Suppress brake/turn for collision tracks
                        if detector_name in ['brake', 'turn']:
                            collision_pairs = getattr(
                                self.behaviour_detectors.get('collision'), 'collision_pairs', {}
                            )
                            suppress_ids = set()
                            for pair_key, pair_data in collision_pairs.items():
                                if isinstance(pair_key, tuple) and len(pair_key) == 2:
                                    suppress_ids.add(pair_key[0])
                                    suppress_ids.add(pair_key[1])
                            suppress_ids.update(collision_track_ids)

                            results = [r for r in results if r.get('track_id', -1) not in suppress_ids]

                        self.behaviour_results[detector_name] = results

                # Restore thresholds
                if original_thresholds is not None:
                    self.scene_analyzer.restore_thresholds(
                        self.behaviour_detectors, original_thresholds
                    )

        physics_time = time.time() - physics_start
        total_time = time.time() - pipeline_start

        # Update stats
        self.pipeline_stats['detection_time'].append(det_time)
        self.pipeline_stats['tracking_time'].append(track_time)
        self.pipeline_stats['physics_time'].append(physics_time)
        self.pipeline_stats['total_pipeline_time'].append(total_time)

        # === 5. RETURN RESULTS ===
        return {
            'frame_id': frame_id,
            'detections': detections,
            'iou_features': {},  # Not used by ByteTrack
            'active_tracks': active_tracks,
            'association_results': {},
            'update_results': {},
            'pipeline_timing': {
                'detection_time': det_time,
                'tracking_time': track_time,
                'physics_time': physics_time,
                'ppl_time': self._frame_ppl_compute_time,
                'alarm_decision_time': self._frame_alarm_decision_time,
                'collision_detector_time': self._frame_collision_detector_time,
                'total_pipeline_time': total_time
            },
            'pipeline_stats': self._get_pipeline_stats()
        }

    def _match_strack_to_detection(self, strack_bbox, detections: List[Dict]) -> Optional[Dict]:
        """Match STrack bbox ke original YOLO detection via IoU untuk mendapat class_name"""
        best_iou = 0.0
        best_det = None

        for det in detections:
            iou = _compute_iou(strack_bbox, det['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_det = det

        # Require minimum IoU 0.3 for a valid match
        if best_iou >= 0.3:
            return best_det
        return None

    def get_current_tracks(self, confirmed_only: bool = False) -> List[Track]:
        """Get current active tracks — compatible interface"""
        if confirmed_only:
            return [t for t in self.confirmed_tracks.values() if t.state == 'active']
        return [t for t in self.tracks.values() if t.state == 'active']

    def get_track_statistics(self) -> Dict:
        """Get tracking statistics"""
        active = len([t for t in self.tracks.values() if t.state == 'active'])
        ghost = len([t for t in self.tracks.values() if t.state == 'ghost'])

        return {
            'total_tracks_created': self.total_tracks_created,
            'active_tracks': active,
            'ghost_tracks': ghost,
            'confirmed_tracks': len(self.confirmed_tracks),
            'total_associations': self.total_associations,
            'total_id_switches': self.total_id_switches,
            'total_ghost_reidentifications': self.total_ghost_reidentifications,
            'frames_processed': self.frame_count,
            'tracker_type': 'bytetrack'
        }

    def visualize_tracks(self, image: np.ndarray,
                        show_confirmed_only: bool = False,
                        show_trajectories: bool = True,
                        show_ghost_tracks: bool = True) -> np.ndarray:
        """Visualize tracks on frame — compatible with PureSmartHungarianTrackManager"""
        import cv2
        vis_image = image.copy()

        tracks_to_show = self.get_current_tracks(confirmed_only=show_confirmed_only)

        for track in tracks_to_show:
            color = self._get_track_color(track.track_id)

            bbox = track.current_detection['bbox']
            x1, y1, x2, y2 = bbox

            thickness = 3 if track.track_id in self.confirmed_tracks else 2
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

            # Label
            confidence = track.current_detection.get('confidence', 0.8)
            class_name = track.current_detection.get('class_name', '?')
            cls_display = class_name.upper() if class_name not in ('?', 'unknown') else '?'
            label = f"[{cls_display}] #{track.track_id} ({confidence:.2f}) H:{track.hits}"

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 5),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Trajectory
            if show_trajectories and len(track.history) > 1:
                centers = [det['center'] for det in track.history[-10:]]
                for i in range(1, len(centers)):
                    pt1 = (int(centers[i-1][0]), int(centers[i-1][1]))
                    pt2 = (int(centers[i][0]), int(centers[i][1]))
                    cv2.line(vis_image, pt1, pt2, color, 2)
                    cv2.circle(vis_image, pt2, 3, color, -1)

        return vis_image

    @staticmethod
    def _get_track_color(track_id: int):
        """Generate consistent color for track ID"""
        np.random.seed(track_id)
        return tuple(np.random.randint(50, 255, 3).tolist())

    def _get_pipeline_stats(self) -> Dict:
        """Get pipeline performance stats"""
        stats = {}
        for key, times in self.pipeline_stats.items():
            if len(times) > 0:
                stats[f'avg_{key}'] = np.mean(times[-100:])
                stats[f'last_{key}'] = times[-1]
        return stats
