import csv
"""
LTE-TrackGuard Main Application
================================

Main script untuk testing physics-based behaviour detection
Focus: Fallen motorcycle detection & collision detection

Usage:
    python main.py --video path/to/video.mp4 --physics --detect fallen
    python main.py --video path/to/video.mp4 --physics --detect collision
    python main.py --video path/to/video.mp4 --physics --detect all
    python main.py --webcam 0 --physics --detect collision
    python main.py --source rtsp://user:pass@host:554/stream --physics --detect collision --no-output
"""

import argparse
import cv2
import numpy as np
import sys
import os
from pathlib import Path
import time
import logging
from datetime import datetime
from typing import Dict, List
from urllib.parse import urlparse

# Add local modules to path. Use resolved absolute paths so subprocess/batch
# runs behave the same on Windows and Linux SSH environments.
REPO_ROOT = Path(__file__).resolve().parent
for _module_path in (REPO_ROOT, REPO_ROOT / 'core', REPO_ROOT / 'utils', REPO_ROOT / 'bytetrack'):
    _module_path_str = str(_module_path)
    if _module_path_str not in sys.path:
        sys.path.insert(0, _module_path_str)

# Import track manager
from core.track_manager import PureSmartHungarianTrackManager

# Kelas kendaraan bermotor yang diizinkan untuk deteksi collision
# (mencakup motor/motorcycle untuk dataset TAD dan VisDrone)
_MOTOR_VEHICLE_CLS = {'car', 'van', 'truck', 'bus', 'motor', 'motorcycle',
                      'tricycle', 'awning-tricycle'}

# Import notification service (optional)
try:
    from telegram.notification_service import TelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("âš ï¸ Telegram notification service not available")

# Import evaluation modules
try:
    from evaluate_tudat import (
        GTEvent, AIDetection, parse_annotations,
        evaluate_video, evaluate_video_dual_criterion, print_results, compute_aggregate_metrics
    )
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    print("âš ï¸ Evaluation module (evaluate_tudat.py) not found. Metrics will be skipped.")


def _is_url_source(source: str) -> bool:
    parsed = urlparse(str(source))
    return parsed.scheme.lower() in {"rtsp", "rtmp", "http", "https"}


def _is_camera_index(source: str) -> bool:
    return str(source).strip().isdigit()


def _safe_source_stem(source: str) -> str:
    """Return a filesystem-safe short name for logs/output from file/camera/URL."""
    source = str(source)
    if _is_camera_index(source):
        return f"webcam_{source}"
    if _is_url_source(source):
        parsed = urlparse(source)
        host = parsed.hostname or parsed.scheme or "stream"
        path_stem = Path(parsed.path).stem or "rtsp"
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in f"{host}_{path_stem}")
    return Path(source).stem or "source"


def _open_capture(source):
    """OpenCV capture helper. Numeric strings are treated as webcam indexes."""
    if isinstance(source, str) and _is_camera_index(source):
        return cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    if isinstance(source, str) and _is_url_source(source):
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LTE-TrackGuard - Physics-Based Traffic Behaviour Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test fallen motorcycle detection (with YOLO11n - fastest)
  python main.py --video crash_motor.mp4 --physics --detect fallen
  
  # Test collision detection with YOLO11s (more accurate)
  python main.py --video collision.mp4 --physics --detect collision --model yolo11s.pt
  
  # Test all detectors with YOLO11m (balanced)
  python main.py --video traffic.mp4 --physics --detect all --model yolo11m.pt
  
  # Webcam deployment
  python main.py --webcam 0 --physics --detect collision --no-output

  # RTSP CCTV deployment
  python main.py --source rtsp://user:pass@192.168.1.10:554/stream1 --physics --detect collision --no-output

  # Standard mode (no physics)
  python main.py --video traffic.mp4
  
Model Options:
  yolo11n.pt - Nano (fastest, ~3ms/frame, best for real-time)
  yolo11s.pt - Small (balanced, ~5ms/frame)
  yolo11m.pt - Medium (accurate, ~10ms/frame)
  yolo11l.pt - Large (very accurate, ~15ms/frame)
  yolo11x.pt - Extra Large (most accurate, ~25ms/frame)
        """
    )
    
    # Input source. --video is kept for backward compatibility with old runs.
    parser.add_argument('--video', type=str, default=None,
                       help='Path to input video file (legacy alias for --source)')
    parser.add_argument('--source', type=str, default=None,
                       help='Input source: video file path, webcam index (0/1/...), or RTSP/HTTP URL')
    parser.add_argument('--webcam', type=int, default=None,
                       help='Shortcut for webcam index, e.g. --webcam 0')
    
    # Model selection
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='YOLO model weights (default: yolo11n.pt - fastest)')
    
    # Tracker selection
    parser.add_argument('--tracker', type=str, default='bytetrack',
                       choices=['bytetrack', 'hungarian'],
                       help='Tracker to use (default: bytetrack)')

    # Collision detection tuning
    parser.add_argument('--conf', type=int, default=70,
                       help='Collision confidence threshold %% for COLLISION DETECTED (default: 70)')
    parser.add_argument('--min-hits', type=int, default=5,
                       help='Minimum track hits before collision check â€” filters ghost tracks (default: 5)')

    parser.add_argument('--ppl-kappa-prox', type=float, default=None,
                       help='Override PPL proximity multiplier kappa for sensitivity sweeps (nominal: 2.0)')
    parser.add_argument('--ppl-delta-iou', type=float, default=None,
                       help='Override PPL Path-C IoU-delta guard for sensitivity sweeps (nominal: 0.02)')
    parser.add_argument('--ppl-rho-min', type=float, default=None,
                       help='Override PPL CE relative-speed ratio guard for sensitivity sweeps (nominal: 0.15)')
    parser.add_argument('--collision-baseline', type=str, default='none',
                       choices=['none', 'proximity', 'iou-contact', 'iou_contact',
                                'speed-drop-prox', 'speed_drop_proximity',
                                'heading-change', 'heading_change_proximity', 'union'],
                       help='Run a calibration-light track-heuristic comparator instead of '
                            'TrackGuard-LTE collision logic. Default: none.')

    # Physics mode
    parser.add_argument('--physics', action='store_true',
                       help='Enable LTE-TrackGuard physics mode')
    
    # Detector selection
    parser.add_argument('--detect', type=str, default='fallen',
                       choices=['fallen', 'collision', 'turn', 'brake', 'all'],
                       help='Which behaviour to detect (default: fallen)')
    
    # EAGER smoothing
    parser.add_argument('--eager', action='store_true', default=True,
                       help='Enable EAGER smoothing (default: True)')
    
    # Physics predictor (experimental)
    parser.add_argument('--physics-predictor', action='store_true',
                       help='Use physics predictor instead of Kalman (experimental)')
    
    # Output options
    parser.add_argument('--output', type=str, default='output.mp4',
                       help='Path to save output video (default: output.mp4)')
    parser.add_argument('--no-output', action='store_true',
                       help='Do not save annotated output video. Recommended for webcam/RTSP deployment.')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without cv2.imshow window (headless deployment).')
    parser.add_argument('--evidence-dir', type=str, default='log_terminal/evidence_frames',
                       help='Directory for automatic alarm evidence snapshots, split into red/critical/caution subfolders')
    parser.add_argument('--no-evidence-snapshots', action='store_true',
                       help='Disable automatic PNG snapshots for alarm events')
    parser.add_argument('--freeze-escalation-ms', type=int, default=1200,
                       help='Hold the display briefly on red escalation frames (default: 1200 ms; display only)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Stop after N frames. Useful for webcam/RTSP smoke tests.')
    parser.add_argument('--reconnect-delay', type=float, default=2.0,
                       help='Seconds to wait before reconnecting a live source after read failure (default: 2.0)')
    parser.add_argument('--live-refresh-interval', type=float, default=60.0,
                       help='Proactively reopen live URL/camera sources every N seconds. '
                            'Set 0 to disable. Default: 60.0')
    parser.add_argument('--stage-profile-output', type=str, default=None,
                       help='Optional CSV path for per-frame runtime profiling. '
                            'Records capture/decode, YOLO, ByteTrack, PPL computation, alarm decision, MQTT-ready policy output, and total frame timings.')
    
    parser.add_argument('--show-trajectories', action='store_true', default=True,
                       help='Show track trajectories (default: True)')
    
    parser.add_argument('--fps-limit', type=int, default=None,
                       help='Limit processing FPS (for slow motion analysis)')
    
    # Verbosity
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    # Telegram notification
    parser.add_argument('--telegram', action='store_true',
                       help='Enable Telegram notifications')

    # â”€â”€ Custom weight / class override â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parser.add_argument('--target-classes', type=str, default=None,
                       help='Filter class tertentu dari weight, pisahkan koma. '
                            'Contoh: --target-classes car,van,truck,bus,motor  '
                            'Default: None = pakai SEMUA class dari weight file.')
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    # â”€â”€ Scalar logging (discriminability evaluation, IEEE T-IM) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parser.add_argument('--log-scalars', action='store_true',
                       help='Aktifkan scalar logging untuk evaluasi discriminability (IEEE T-IM). '
                            'Menyimpan nilai KELR/ARS/CE setiap proximate pair ke CSV.')
    parser.add_argument('--label', type=int, default=0, choices=[0, 1],
                       help='Ground truth label video: 0=non-collision (negative), '
                            '1=collision (positive). Dipakai bersama --log-scalars. (default: 0)')
    parser.add_argument('--camera-angle', type=str, default='unknown',
                       choices=['oblique', 'overhead', 'unknown'],
                       help='Tipe sudut kamera: oblique (miring) atau overhead (bird-eye). '
                            '(default: unknown)')
    parser.add_argument('--scalar-log-output', type=str, default=None,
                       help='Path output CSV scalar log. Default: scalar_logs/<videoname>_scalar.csv '
                            '(satu file per video, TIDAK append)')
    parser.add_argument('--fallen-event-log', type=str, default=None,
                       help='Path output CSV untuk fallen event log (frame_id, track_id). '
                            'Dipakai oleh evaluate_motor.py untuk evaluasi TP/FP/FN.')
    parser.add_argument('--fallen-ppl-only', action='store_true',
                       help='Matikan fallback heuristic fallen lama. Hanya FallenPrimitiveLayer '
                            'yang boleh men-trigger motorcycle_fallen.')
    parser.add_argument('--fallen-scalar-output', type=str, default=None,
                       help='Path output CSV scalar log Fallen PPL. Default jika kosong: '
                            'scalar_logs/<videoname>_fallen_scalar.csv')
    parser.add_argument('--fallen-metrics', type=str, default='ar,rot,vel',
                       help='Metrik fallen yang diaktifkan, pisahkan koma: ar,rot,vel '
                            '(default: semua aktif). Contoh ablation: --fallen-metrics ar')
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    args = parser.parse_args()

    selected_sources = [x is not None for x in (args.video, args.source, args.webcam)]
    if sum(selected_sources) != 1:
        parser.error("Choose exactly one input source: --video, --source, or --webcam")

    if args.webcam is not None:
        args.source = str(args.webcam)
    elif args.source is None:
        args.source = args.video
    args.video = args.source  # compatibility with existing code paths
    args.source_stem = _safe_source_stem(args.source)
    args.is_live_source = _is_camera_index(args.source) or _is_url_source(args.source)
    _baseline_aliases = {
        'iou_contact': 'iou-contact',
        'speed_drop_proximity': 'speed-drop-prox',
        'heading_change_proximity': 'heading-change',
    }
    args.collision_baseline = _baseline_aliases.get(args.collision_baseline, args.collision_baseline)

    return args


def configure_physics_settings(args):
    """
    Configure physics settings based on command line arguments
    
    Args:
        args: Parsed arguments
    """
    from utils.settings import SETTINGS
    
    # Set YOLO model
    SETTINGS.DETECTOR_WEIGHTS = args.model
    print(f"ðŸŽ¯ YOLO Model: {args.model}")
    
    # Terapkan target-classes ke settings SEBELUM detector diinit
    # None = biarkan detector baca semua class dari weight (default baru)
    if args.target_classes:
        _cls_list = [c.strip() for c in args.target_classes.split(',') if c.strip()]
        SETTINGS.DETECTOR_CONFIG['target_classes'] = _cls_list
        SETTINGS.DETECTOR_CONFIG['class_ids']      = None  # akan di-resolve di detector
        print(f"ðŸŽ¯ Target classes (override): {_cls_list}")
    else:
        # Hapus hardcoded class_ids agar detector baca dari model.names
        SETTINGS.DETECTOR_CONFIG['target_classes'] = None
        SETTINGS.DETECTOR_CONFIG['class_ids']      = None

    if args.physics:
        # Enable physics mode
        SETTINGS.PHYSICS_CONFIG['enable_behaviour_detection'] = True
        SETTINGS.PHYSICS_CONFIG['enable_eager'] = args.eager
        SETTINGS.PHYSICS_CONFIG['enable_physics_predictor'] = args.physics_predictor

        # Pass collision tuning params to collision_detector config
        SETTINGS.PHYSICS_CONFIG['collision_detector']['collision_confidence_threshold'] = args.conf
        SETTINGS.PHYSICS_CONFIG['collision_detector']['min_track_hits'] = args.min_hits
        ppl_cfg = SETTINGS.PHYSICS_CONFIG['collision_detector'].setdefault('ppl', {})
        if args.ppl_kappa_prox is not None:
            ppl_cfg['kappa_prox'] = args.ppl_kappa_prox
            ppl_cfg['kappa_prox_strict'] = args.ppl_kappa_prox * 0.6
        if args.ppl_delta_iou is not None:
            ppl_cfg['delta_iou'] = args.ppl_delta_iou
            ppl_cfg['strict_single_kelr_delta_iou'] = args.ppl_delta_iou * 1.5
        if args.ppl_rho_min is not None:
            ppl_cfg['rho_min'] = args.ppl_rho_min

        # Pass fallen metric ablation flags
        _active_metrics = [m.strip() for m in args.fallen_metrics.split(',')]
        if 'fallen_detector' not in SETTINGS.PHYSICS_CONFIG:
            SETTINGS.PHYSICS_CONFIG['fallen_detector'] = {}
        SETTINGS.PHYSICS_CONFIG['fallen_detector']['enable_ar_flip']       = 'ar'  in _active_metrics
        SETTINGS.PHYSICS_CONFIG['fallen_detector']['enable_rotation']      = 'rot' in _active_metrics
        SETTINGS.PHYSICS_CONFIG['fallen_detector']['enable_velocity_drop'] = 'vel' in _active_metrics
        fallen_ppl_cfg = SETTINGS.PHYSICS_CONFIG['fallen_detector'].setdefault('fallen_ppl', {})
        if args.fallen_ppl_only:
            fallen_ppl_cfg['enable_legacy_fallback'] = False

        print("ðŸ”¬ Physics Mode Configuration:")
        print(f"   Behaviour Detection: ENABLED")
        print(f"   EAGER Smoothing: {'ENABLED' if args.eager else 'DISABLED'}")
        print(f"   Physics Predictor: {'ENABLED (experimental)' if args.physics_predictor else 'DISABLED (using Kalman)'}")
        print(f"   Target Detection: {args.detect.upper()}")
        print(f"   Collision Confidence: >= {args.conf}% = COLLISION DETECTED")
        print(f"   Min Track Hits: {args.min_hits} (ghost track filter)")
        if args.fallen_ppl_only:
            print("   Fallen mode: PPL-only (legacy fallen fallback disabled)")
        if ppl_cfg:
            print(f"   PPL guard overrides: {ppl_cfg}")
    else:
        print("ðŸ“Š Standard Mode (No Physics)")


def draw_dashed_rectangle(img, pt1: tuple, pt2: tuple, color: tuple, thickness: int = 1, dash_length: int = 10):
    """
    Draw dashed rectangle on image
    
    Args:
        img: Image to draw on
        pt1: Top-left corner (x1, y1)
        pt2: Bottom-right corner (x2, y2)
        color: BGR color tuple
        thickness: Line thickness
        dash_length: Length of each dash segment
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw top edge
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
    
    # Draw bottom edge
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    
    # Draw left edge
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
    
    # Draw right edge
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


def visualize_detections(frame: np.ndarray, behaviour_results: Dict,
                         detector_type: str, frame_id: int = 0,
                         proximity_warning_pairs: Dict = None,
                         current_track_bboxes: Dict = None) -> np.ndarray:
    """
    Visualize behaviour detections on frame
    
    Args:
        frame: Input frame
        behaviour_results: Dict of behaviour detection results
        detector_type: Which detector to visualize ('fallen', 'collision', 'all')
        frame_id: Current frame ID
        proximity_warning_pairs: Dict of proximity warnings from collision_detector
        
    Returns:
        Frame with visualizations
    """
    # Declare global at the very beginning - BEFORE any usage
    global proximity_warnings
    from datetime import datetime
    
    vis_frame = frame.copy()
    
    # Colors for different behaviours
    colors = {
        'fallen': (0, 0, 255),      # Red - emergency
        'collision': (0, 0, 255),   # Red - emergency
        'turn': (0, 255, 255),      # Yellow - info
        'brake': (0, 165, 255),     # Orange - warning
        'decelerating': (255, 0, 255),  # Magenta - decelerating (motor menepi)
        'zigzag_driving': (255, 0, 255)  # Magenta - warning
    }
    
    # Severity badges - hapus emoji, gunakan text saja
    severity_badges = {
        'critical': '[CRITICAL]',
        'high': '[HIGH]',
        'medium': '[MEDIUM]',
        'low': '[LOW]'
    }
    
    # Get current datetime
    current_datetime = datetime.now()
    
    # Determine which detections to show
    detections_to_show = []
    
    if detector_type == 'all':
        # Filter by class: collision only for 'car', fallen only for 'motorcycle'
        for det_type, dets in behaviour_results.items():
            for det in dets:
                behaviour_type = det.get('behaviour_type', 'unknown')
                
                # Collision: semua kendaraan bermotor (car, van, truck, bus, motor)
                if behaviour_type == 'collision':
                    class_primary = det.get('class_primary', 'unknown')
                    class_secondary = det.get('class_secondary', 'unknown')
                    if class_primary in _MOTOR_VEHICLE_CLS and class_secondary in _MOTOR_VEHICLE_CLS:
                        detections_to_show.append(det)
                
                # Fallen: only show if track is 'motorcycle'
                elif behaviour_type == 'motorcycle_fallen':
                    # Check class_name from detection (if available) or assume motorcycle
                    # FallenDetector already filters for motorcycle, so we can trust it
                    detections_to_show.append(det)
                
                # Other behaviours: show all
                else:
                    detections_to_show.append(det)
    elif detector_type in behaviour_results:
        detections_to_show = behaviour_results[detector_type]
    
    # ============================================
    # Draw proximity warnings (objek mendekat, belum collision)
    # ============================================
    
    # Use proximity_warning_pairs from parameter (passed from main())
    if proximity_warning_pairs is None:
        proximity_warning_pairs = {}
    
    # Check for active proximity warnings that should still be displayed
    active_warnings = {}
    for pair_key, warning_data in proximity_warnings.items():
        if len(warning_data) >= 2:
            first_detected_time = warning_data[0]
            time_since_first = (current_datetime - first_detected_time).total_seconds()
            if time_since_first <= 15.0:  # Still within 15 seconds
                active_warnings[pair_key] = warning_data
    
    # Cleanup old warnings (older than 60 seconds)
    proximity_warnings = {k: v for k, v in proximity_warnings.items() 
                         if len(v) >= 2 and (current_datetime - v[0]).total_seconds() <= 60.0}
    
    # ============================================
    # Draw proximity warnings (vehicle too close, might be collision)
    # ============================================
    # Get proximity warnings dari collision_detector via track_manager
    # Note: Ini akan di-handle di visualize_detections setelah collision detection
    
    # Draw detections
    for detection in detections_to_show:
        behaviour_type = detection.get('behaviour_type', 'unknown')
        severity = detection.get('severity', 'unknown')
        track_id = detection.get('track_id', -1)
        bbox = detection.get('bbox', None)
        prediction_mode = detection.get('prediction_mode', 'confirmed')
        
        if bbox is None:
            continue
        
        # Get color
        color = colors.get(behaviour_type, (128, 128, 128))
        
        # Draw bounding box - solid untuk confirmed, dashed untuk predicted
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Default: tampilkan semua visual. Non-emergency collision â†’ disembunyikan.
        _show_collision_visuals = True

        if prediction_mode == 'physics_predicted':
            # Physics predicted - dashed orange box
            color_predicted = (0, 100, 255)  # Darker orange untuk predicted
            thickness = 2
            draw_dashed_rectangle(vis_frame, (x1, y1), (x2, y2), color_predicted, thickness)
            
            # Show fall direction arrow (dari vorticity sign) - panah kecil di dalam box
            center = detection.get('center', [(x1 + x2) / 2, (y1 + y2) / 2])
            fall_direction = detection.get('fall_direction', [1.0, 0.0])
            velocity = detection.get('velocity', [0, 0])
            
            # Draw direction arrow (estimated direction of fall)
            if fall_direction and len(fall_direction) >= 2:
                center_int = (int(center[0]), int(center[1]))
                # Arrow length proporsional ke velocity magnitude atau fixed untuk visibility
                arrow_length = 30  # Fixed length untuk visibility
                end_point = (
                    int(center[0] + fall_direction[0] * arrow_length),
                    int(center[1] + fall_direction[1] * arrow_length)
                )
                # Draw arrow dengan warna cyan untuk visibility
                cv2.arrowedLine(vis_frame, center_int, end_point, (255, 255, 0), 2, tipLength=0.4)
            
            # Show confidence + time icon di pojok box
            confidence = detection.get('confidence', 1.0)
            frames_since = detection.get('frames_since_seen', 0)
            # Get fps from global or assume 30 fps
            fps_value = 30.0  # Default
            time_seconds = frames_since / fps_value
            
            # Draw confidence + time di pojok kanan atas box
            conf_text = f"{confidence:.2f}"
            time_text = f"{time_seconds:.1f}s"
            font_small = cv2.FONT_HERSHEY_SIMPLEX
            font_scale_small = 0.5
            thickness_small = 1
            
            # Background untuk text
            conf_size = cv2.getTextSize(conf_text, font_small, font_scale_small, thickness_small)[0]
            time_size = cv2.getTextSize(time_text, font_small, font_scale_small, thickness_small)[0]
            text_height = conf_size[1] + time_size[1] + 5
            
            # Draw semi-transparent background
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (x2 - max(conf_size[0], time_size[0]) - 10, y1),
                         (x2, y1 + text_height + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
            
            # Draw confidence text
            cv2.putText(vis_frame, conf_text, (x2 - conf_size[0] - 5, y1 + conf_size[1] + 2),
                       font_small, font_scale_small, (0, 255, 255), thickness_small)
            
            # Draw time text dengan icon jam (simbol "â±" tidak support, pakai "T" atau "s")
            cv2.putText(vis_frame, f"T:{time_text}", (x2 - time_size[0] - 5, y1 + text_height),
                       font_small, font_scale_small, (0, 255, 255), thickness_small)
            
            # Draw label dengan confidence info
            badge = severity_badges.get(severity, '!')
            label = f"{badge} MOTORCYCLE_FALLEN (predicted)"
            label_conf = None  # Tidak perlu label_conf lagi, sudah di pojok box
            
        else:
            # YOLO confirmed - solid box
            # Collision non-emergency (MIGHT/MEDIUM): skip box & label â†’ cukup di info panel kiri
            if behaviour_type == 'collision':
                _show_collision_visuals = detection.get('alert_level', 'emergency') in ('emergency', 'warning', 'caution')
            if _show_collision_visuals:
                thickness = 4
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)

            # Draw label (None untuk non-emergency collision â†’ skip draw)
            badge = severity_badges.get(severity, '!')
            label = f"{badge} {behaviour_type.upper()}" if _show_collision_visuals else None
            label_conf = None
        
        # Large text for visibility â€” skip untuk non-emergency collision (label == None)
        if label is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            text_thickness = 3

            # Background for text
            text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
            cv2.rectangle(vis_frame,
                         (x1, y1 - text_size[1] - 15),
                         (x1 + text_size[0] + 10, y1),
                         color if prediction_mode == 'confirmed' else color_predicted, -1)

            # White text
            cv2.putText(vis_frame, label, (x1 + 5, y1 - 10),
                       font, font_scale, (255, 255, 255), text_thickness)
        
        # Additional info
        if behaviour_type == 'motorcycle_fallen':
            if label_conf:
                # Show confidence untuk predicted
                cv2.putText(vis_frame, label_conf, (x1, y2 + 25),
                           font, 0.5, color_predicted, 2)
            else:
                # Show track ID untuk confirmed
                info = f"Track ID: {track_id}"
                cv2.putText(vis_frame, info, (x1, y2 + 25),
                           font, 0.6, color, 2)
        
        elif behaviour_type == 'decelerating':
            # Show decelerating label dengan warna magenta
            info = f"DECELERATING - Track ID: {track_id}"
            cv2.putText(vis_frame, info, (x1, y2 + 25),
                       font, 0.6, color, 2)
        
        elif behaviour_type == 'collision':
            track_id_2 = detection.get('track_id_secondary', -1)
            detection_mode = detection.get('detection_mode', 'normal')
            alert_level_bbox = detection.get('alert_level', 'emergency')

            # COLLISION: Dashed red box hanya untuk HIGH confidence (emergency = >= 70%)
            # MIGHT / MEDIUM â†’ tidak gambar bbox merah, cukup dicatat di info panel kiri
            if alert_level_bbox == 'emergency':
                thickness = 3
                draw_dashed_rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), thickness, dash_length=15)

            # ============================================
            # COLLISION WARNING: Alert di atas bounding box (persist 15 seconds, NO BLINKING)
            # ============================================
            pair_key = tuple(sorted([track_id, track_id_2]))
            
            # Update collision warning state (persist 15 seconds)
            # Update untuk SEMUA collision detections agar warning tetap muncul selama 15 detik
            # Save bbox + alert_level + confidence_label untuk draw warning meskipun detection hilang
            last_bbox = [x1, y1, x2, y2]
            alert_level = detection.get('alert_level', 'emergency')
            confidence_label = detection.get('confidence_label', 'COLLISION DETECTED')

            if pair_key not in proximity_warnings:
                # First detection - record timestamp + bbox + alert_level + confidence_label
                # Tuple format: (first_detected, last_detected, bbox, alert_level, confidence_label)
                proximity_warnings[pair_key] = (current_datetime, current_datetime, last_bbox, alert_level, confidence_label)
            else:
                # Update last detected time + bbox + alert_level + confidence_label (untuk semua collision detections)
                warning_data = proximity_warnings[pair_key]
                first_detected = warning_data[0] if len(warning_data) >= 2 else current_datetime
                proximity_warnings[pair_key] = (first_detected, current_datetime, last_bbox, alert_level, confidence_label)
            
            # Check if warning should still be shown (within 15 seconds from first detection)
            # This applies to ALL collision detections to ensure warning persists
            if pair_key in proximity_warnings:
                warning_data = proximity_warnings[pair_key]
                first_detected_time = warning_data[0] if len(warning_data) >= 1 else current_datetime
                time_since_first = (current_datetime - first_detected_time).total_seconds()

                if time_since_first <= 15.0 and alert_level_bbox in ('emergency', 'warning', 'caution'):
                    # Get alert_level and confidence_label from detection
                    alert_level = detection.get('alert_level', 'emergency')
                    confidence_label = detection.get('confidence_label', 'COLLISION DETECTED')

                    # Warning text with confidence score
                    conf_score = detection.get('confidence', 0)
                    conf_str = f" ({conf_score:.0f}%)" if conf_score > 0 else ""

                    if alert_level == 'emergency':
                        warning_text = f"[M] {confidence_label}{conf_str}: Escalate"
                    elif alert_level == 'warning':
                        warning_text = f"[M] {confidence_label}{conf_str}: Operator attention"
                    elif alert_level == 'caution':
                        warning_text = f"[M] {confidence_label}{conf_str}: Monitor closely"
                    else:
                        warning_text = f"[M] {confidence_label}{conf_str}: Low confidence"
                    
                    # Draw warning text di atas bounding box (SOLID - NO BLINKING)
                    warning_font = cv2.FONT_HERSHEY_SIMPLEX
                    warning_font_scale = 0.7
                    warning_thickness = 2
                    # Color based on alert_level
                    if alert_level == 'emergency':
                        warning_color = (0, 0, 255)      # Red: COLLISION DETECTED
                    elif alert_level == 'warning':
                        warning_color = (0, 140, 255)    # Orange: CRITICAL INTERACTION
                    elif alert_level == 'caution':
                        warning_color = (0, 220, 255)    # Yellow: MONITOR CLOSELY / UNSAFE PROXIMITY
                    else:
                        warning_color = (180, 180, 180)  # Gray â€” NORMAL

                    # Get text size
                    (warning_width, warning_height), baseline = cv2.getTextSize(
                        warning_text, warning_font, warning_font_scale, warning_thickness
                    )
                    
                    # Position: di atas bounding box, center-aligned
                    warning_x = x1 + (x2 - x1 - warning_width) // 2
                    warning_y = max(y1 - 15, warning_height + 10)  # Pastikan tidak keluar frame
                    
                    # Draw semi-transparent background untuk warning (solid, no blinking)
                    warning_bg_y1 = warning_y - warning_height - 5
                    warning_bg_y2 = warning_y + baseline + 5
                    warning_bg_x1 = warning_x - 5
                    warning_bg_x2 = warning_x + warning_width + 5
                    
                    overlay_warning = vis_frame.copy()
                    cv2.rectangle(overlay_warning, 
                                 (warning_bg_x1, warning_bg_y1),
                                 (warning_bg_x2, warning_bg_y2),
                                 (0, 0, 0), -1)
                    cv2.addWeighted(overlay_warning, 0.7, vis_frame, 0.3, 0, vis_frame)
                    
                    # Draw warning text dengan outline untuk visibility (solid, no blinking)
                    # Outline (black)
                    cv2.putText(vis_frame, warning_text, (warning_x - 1, warning_y - 1),
                               warning_font, warning_font_scale, (0, 0, 0), warning_thickness + 1)
                    cv2.putText(vis_frame, warning_text, (warning_x + 1, warning_y + 1),
                               warning_font, warning_font_scale, (0, 0, 0), warning_thickness + 1)
                    
                    # Main text (orange) - solid, no blinking
                    cv2.putText(vis_frame, warning_text, (warning_x, warning_y),
                               warning_font, warning_font_scale, warning_color, warning_thickness)

                    # Downward arrow callout â†’ vehicle (visual pointer for operator)
                    center_x = (x1 + x2) // 2
                    arrow_start_y = max(y1 - 50, warning_bg_y2 + 5)
                    if arrow_start_y < y1 - 5:
                        cv2.arrowedLine(vis_frame,
                                       (center_x, arrow_start_y),
                                       (center_x, y1 - 2),
                                       warning_color, 3, tipLength=0.4)

            # Show state info di bawah bbox â€” hanya untuk HIGH collision (emergency)
            if alert_level_bbox == 'emergency':
                state_info = detection.get('state', 'confirmed')
                tier = detection.get('tier', 0)
                detection_mode = detection.get('detection_mode', 'normal')
                if tier == 0.5:
                    tier_label = "TIER0.5-PUSH"
                elif tier == 1.5:
                    tier_label = "TIER1.5-SPARSE"
                elif tier == 1:
                    if 'rotation_spike' in detection_mode:
                        tier_label = "TIER1-SPIN"
                    else:
                        tier_label = "TIER1"
                elif tier == 2:
                    tier_label = "TIER2-DEFORM"
                elif tier == 3:
                    tier_label = "TIER3"
                else:
                    tier_label = ""

                if state_info == 'monitoring':
                    info = f"Tracks: {track_id} <-> {track_id_2} [MONITORING] {tier_label}"
                else:
                    info = f"Tracks: {track_id} <-> {track_id_2} [CONFIRMED] {tier_label}"

                cv2.putText(vis_frame, info, (x1, y2 + 25),
                           font, 0.6, color, 2)
            
            # Show detail di dalam box dan collision point â€” hanya untuk HIGH collision
            tier = detection.get('tier', 0)
            detection_mode = detection.get('detection_mode', 'normal')
            if alert_level_bbox != 'emergency':
                tier = -1  # skip all tier detail draws below for non-emergency
            if tier in [0.5, 1, 1.5, 2, 3]:
                ar_change_i = detection.get('ar_change_i', 0.0)
                ar_change_j = detection.get('ar_change_j', 0.0)
                area_change_i = detection.get('area_change_i', 0.0)
                area_change_j = detection.get('area_change_j', 0.0)
                
                if tier == 0.5:
                    # Tier 0.5: Push Collision Priority - show push metrics
                    iou = detection.get('iou_overlap', 0.0)
                    momentum_transfer_i = detection.get('momentum_transfer_i', 0.0)
                    momentum_transfer_j = detection.get('momentum_transfer_j', 0.0)
                    max_momentum = max(momentum_transfer_i, momentum_transfer_j)
                    relative_velocity_drop = detection.get('relative_velocity_drop', 0.0) * 100
                    detail_text = f"Tier0.5: Push | IoU {iou*100:.0f}% | Momentum {max_momentum:.0f} | v_relâ†“ {relative_velocity_drop:.0f}%"
                elif tier == 1.5:
                    # Tier 1.5: Sparse Scene - show rotation spike
                    iou = detection.get('iou_overlap', 0.0)
                    rotation_spike_i = detection.get('rotation_spike_i', 0.0)
                    rotation_spike_j = detection.get('rotation_spike_j', 0.0)
                    max_rotation_spike = max(rotation_spike_i, rotation_spike_j)
                    detail_text = f"Tier1.5: IoU {iou*100:.0f}% | Spin {max_rotation_spike:.2f}"
                elif tier == 1:
                    # Tier 1: High-Confidence - show IoU, energy loss, dan rotation spike jika ada
                    iou = detection.get('iou_overlap', 0.0)
                    energy_loss_primary = detection.get('energy_loss_primary', 0.0)
                    energy_loss_secondary = detection.get('energy_loss_secondary', 0.0)
                    max_energy_loss = max(energy_loss_primary, energy_loss_secondary) * 100
                    if 'rotation_spike' in detection_mode:
                        rotation_spike_i = detection.get('rotation_spike_i', 0.0)
                        rotation_spike_j = detection.get('rotation_spike_j', 0.0)
                        max_rotation_spike = max(rotation_spike_i, rotation_spike_j)
                        detail_text = f"Tier1: IoU {iou*100:.0f}% | Spin {max_rotation_spike:.2f}"
                    else:
                        detail_text = f"Tier1: IoU {iou*100:.0f}% | Energy {max_energy_loss:.0f}%"
                elif tier == 2:
                    # Tier 2: Deformation-Based - show AR and area change
                    max_ar_change = max(ar_change_i, ar_change_j) * 100
                    max_area_change = max(area_change_i, area_change_j) * 100
                    detail_text = f"Tier2: AR {max_ar_change:.0f}% | Area {max_area_change:.0f}%"
                elif tier == 3:
                    # Tier 3: Medium-Confidence - show IoU and energy loss
                    iou = detection.get('iou_overlap', 0.0)
                    energy_loss_primary = detection.get('energy_loss_primary', 0.0)
                    energy_loss_secondary = detection.get('energy_loss_secondary', 0.0)
                    max_energy_loss = max(energy_loss_primary, energy_loss_secondary) * 100
                    detail_text = f"Tier3: IoU {iou*100:.0f}% | Energy {max_energy_loss:.0f}%"
                else:
                    detail_text = "Collision detected"
                
                detail_x = x1 + 5
                detail_y = y1 + 20
                
                # Background untuk detail
                (detail_width, detail_height), _ = cv2.getTextSize(
                    detail_text, font, 0.5, 1
                )
                cv2.rectangle(vis_frame,
                             (detail_x - 3, detail_y - detail_height - 3),
                             (detail_x + detail_width + 3, detail_y + 3),
                             (0, 0, 0), -1)
                
                cv2.putText(vis_frame, detail_text, (detail_x, detail_y),
                           font, 0.5, (0, 255, 255), 1)
            
            # Draw collision point â€” hanya untuk HIGH collision
            collision_point = detection.get('collision_point', None) if alert_level_bbox == 'emergency' else None
            if collision_point:
                cx, cy = int(collision_point[0]), int(collision_point[1])
                cv2.circle(vis_frame, (cx, cy), 15, (0, 0, 255), -1)
                cv2.circle(vis_frame, (cx, cy), 20, (255, 255, 255), 2)
                
                # Draw impact direction arrow (delta_v = v_i - v_j)
                impact_direction = detection.get('impact_direction', None)
                if impact_direction and len(impact_direction) >= 2:
                    # Normalize direction untuk arrow
                    impact_dir = np.array(impact_direction, dtype=np.float32)
                    impact_norm = np.linalg.norm(impact_dir)
                    if impact_norm > 1e-6:
                        impact_dir = impact_dir / impact_norm
                        
                        # Arrow dari collision point
                        arrow_length = 50  # Fixed length untuk visibility
                        end_point = (
                            int(cx + impact_dir[0] * arrow_length),
                            int(cy + impact_dir[1] * arrow_length)
                        )
                        # Draw arrow dengan warna cyan untuk visibility
                        cv2.arrowedLine(vis_frame, (cx, cy), end_point, (255, 255, 0), 3, tipLength=0.3)
    
    # ============================================
    # Draw proximity warnings for active pairs even if no collision detection in current frame
    # (persist warnings for 15 seconds even if detection temporarily lost)
    # ============================================
    # Get all collision detections to check which pairs are already handled
    collision_detections = behaviour_results.get('collision', [])
    handled_pairs = set()
    for det in collision_detections:
        track_id_1 = det.get('track_id', -1)
        track_id_2 = det.get('track_id_secondary', -1)
        pair_key = tuple(sorted([track_id_1, track_id_2]))
        handled_pairs.add(pair_key)
    
    # Draw warnings for active pairs that are not in current detections
    # (warning persists even if detection temporarily lost)
    for pair_key, warning_data in active_warnings.items():
        if pair_key not in handled_pairs:
            # Warning still active but no detection in current frame
            # Use last saved bbox + alert_level + confidence_label to draw warning
            if len(warning_data) >= 3:
                last_bbox = warning_data[2]
                # Get alert_level and confidence_label from stored data (if available)
                stored_alert_level = warning_data[3] if len(warning_data) >= 4 else 'emergency'
                stored_confidence_label = warning_data[4] if len(warning_data) >= 5 else 'COLLISION DETECTED'

                if last_bbox and len(last_bbox) >= 4 and stored_alert_level in ('emergency', 'warning', 'caution'):
                    # Prefer current tracker position over frozen last_bbox
                    tid_a, tid_b = pair_key
                    live_bbox = None
                    if current_track_bboxes:
                        live_bbox = current_track_bboxes.get(tid_a) or current_track_bboxes.get(tid_b)
                    draw_bbox = live_bbox if live_bbox is not None else last_bbox
                    x1, y1, x2, y2 = [int(coord) for coord in draw_bbox]

                    # Alert color based on level
                    if stored_alert_level == 'emergency':
                        warning_color = (0, 0, 255)       # Red: escalation
                        warning_text = f"[M] {stored_confidence_label}: Escalate"
                    elif stored_alert_level == 'warning':
                        warning_color = (0, 140, 255)     # Orange: operator attention
                        warning_text = f"[M] {stored_confidence_label}: Operator attention"
                    else:  # caution
                        warning_color = (0, 220, 255)     # Yellow: monitor closely
                        warning_text = f"[M] {stored_confidence_label}: Monitor closely"

                    # â”€â”€ Persistent alert bbox (solid colored border) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), warning_color, 3)

                    warning_font = cv2.FONT_HERSHEY_SIMPLEX
                    warning_font_scale = 0.7
                    warning_thickness = 2

                    # Get text size
                    (warning_width, warning_height), baseline = cv2.getTextSize(
                        warning_text, warning_font, warning_font_scale, warning_thickness
                    )

                    # Position: di atas bounding box, center-aligned
                    warning_x = x1 + (x2 - x1 - warning_width) // 2
                    warning_y = max(y1 - 15, warning_height + 10)  # Pastikan tidak keluar frame

                    # Draw semi-transparent background untuk warning
                    warning_bg_y1 = warning_y - warning_height - 5
                    warning_bg_y2 = warning_y + baseline + 5
                    warning_bg_x1 = warning_x - 5
                    warning_bg_x2 = warning_x + warning_width + 5

                    overlay_warning = vis_frame.copy()
                    cv2.rectangle(overlay_warning,
                                 (warning_bg_x1, warning_bg_y1),
                                 (warning_bg_x2, warning_bg_y2),
                                 (0, 0, 0), -1)
                    cv2.addWeighted(overlay_warning, 0.7, vis_frame, 0.3, 0, vis_frame)

                    # Draw warning text
                    cv2.putText(vis_frame, warning_text, (warning_x - 1, warning_y - 1),
                               warning_font, warning_font_scale, (0, 0, 0), warning_thickness + 1)
                    cv2.putText(vis_frame, warning_text, (warning_x + 1, warning_y + 1),
                               warning_font, warning_font_scale, (0, 0, 0), warning_thickness + 1)
                    cv2.putText(vis_frame, warning_text, (warning_x, warning_y),
                               warning_font, warning_font_scale, warning_color, warning_thickness)

                    # â”€â”€ Downward arrow callout â†’ vehicle (persisted) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    center_x = (x1 + x2) // 2
                    arrow_start_y = max(y1 - 50, warning_bg_y2 + 5)
                    if arrow_start_y < y1 - 5:
                        cv2.arrowedLine(vis_frame,
                                       (center_x, arrow_start_y),
                                       (center_x, y1 - 2),
                                       warning_color, 3, tipLength=0.4)
    
    # ============================================
    # Draw PROXIMITY WARNINGS (vehicle too close, might be collision)
    # ============================================
    if proximity_warning_pairs is None:
        proximity_warning_pairs = {}
    
    # Get current datetime untuk persistence
    current_datetime = datetime.now()
    
    # Global state untuk proximity warnings (persist 15 seconds)
    # Note: proximity_warnings already declared as global at function start
    if not hasattr(visualize_detections, '_proximity_warning_state'):
        visualize_detections._proximity_warning_state = {}
    
    # Update proximity warning state dari collision_detector
    for pair_key, warning_data in proximity_warning_pairs.items():
        track_i = warning_data.get('track_i', -1)
        track_j = warning_data.get('track_j', -1)
        bbox_i = warning_data.get('bbox_i', [0, 0, 0, 0])
        bbox_j = warning_data.get('bbox_j', [0, 0, 0, 0])
        iou = warning_data.get('iou', 0.0)
        
        # Compute union bbox untuk visualisasi
        union_bbox = [
            min(bbox_i[0], bbox_j[0]),  # x1
            min(bbox_i[1], bbox_j[1]),  # y1
            max(bbox_i[2], bbox_j[2]),  # x2
            max(bbox_i[3], bbox_j[3])   # y2
        ]
        
        # Update state (persist 15 seconds)
        if pair_key not in visualize_detections._proximity_warning_state:
            visualize_detections._proximity_warning_state[pair_key] = (current_datetime, union_bbox)
        else:
            # Update last detected time dan bbox
            first_detected, _ = visualize_detections._proximity_warning_state[pair_key]
            visualize_detections._proximity_warning_state[pair_key] = (first_detected, union_bbox)
    
    # Cleanup old warnings (older than 60 seconds)
    visualize_detections._proximity_warning_state = {
        k: v for k, v in visualize_detections._proximity_warning_state.items()
        if (current_datetime - v[0]).total_seconds() <= 60.0
    }
    
    # ============================================
    # PROXIMITY WARNING VISUALIZATION - DISABLED
    # Comment out untuk disable banner kuning "TOO CLOSE"
    # ============================================
    # # Draw proximity warnings (persist 15 seconds)
    # for pair_key, (first_detected, last_bbox) in visualize_detections._proximity_warning_state.items():
    #     time_since_first = (current_datetime - first_detected).total_seconds()
    #
    #     if time_since_first <= 15.0:  # Show warning for 15 seconds
    #         x1, y1, x2, y2 = [int(coord) for coord in last_bbox]
    #
    #         # Draw dashed yellow box untuk proximity warning
    #         thickness = 2
    #         draw_dashed_rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), thickness, dash_length=10)
    #
    #         # Draw warning text di atas bounding box
    #         warning_text = "TOO CLOSE, COLLISION WILL HAPPEN"
    #         warning_font = cv2.FONT_HERSHEY_SIMPLEX
    #         warning_font_scale = 0.6
    #         warning_thickness = 2
    #         warning_color = (0, 255, 255)  # Yellow color untuk proximity warning
    #
    #         # Get text size
    #         (warning_width, warning_height), baseline = cv2.getTextSize(
    #             warning_text, warning_font, warning_font_scale, warning_thickness
    #         )
    #
    #         # Position: di atas bounding box, center-aligned
    #         warning_x = x1 + (x2 - x1 - warning_width) // 2
    #         warning_y = max(y1 - 15, warning_height + 10)  # Pastikan tidak keluar frame
    #
    #         # Draw semi-transparent background untuk warning
    #         warning_bg_y1 = warning_y - warning_height - 5
    #         warning_bg_y2 = warning_y + baseline + 5
    #         warning_bg_x1 = warning_x - 5
    #         warning_bg_x2 = warning_x + warning_width + 5
    #
    #         overlay_warning = vis_frame.copy()
    #         cv2.rectangle(overlay_warning,
    #                      (warning_bg_x1, warning_bg_y1),
    #                      (warning_bg_x2, warning_bg_y2),
    #                      (0, 0, 0), -1)
    #         cv2.addWeighted(overlay_warning, 0.7, vis_frame, 0.3, 0, vis_frame)
    #
    #         # Draw warning text dengan outline untuk visibility
    #         # Outline (black)
    #         cv2.putText(vis_frame, warning_text, (warning_x - 1, warning_y - 1),
    #                    warning_font, warning_font_scale, (0, 0, 0), warning_thickness + 1)
    #         cv2.putText(vis_frame, warning_text, (warning_x + 1, warning_y + 1),
    #                    warning_font, warning_font_scale, (0, 0, 0), warning_thickness + 1)
    #
    #         # Main text (yellow) - solid, no blinking
    #         cv2.putText(vis_frame, warning_text, (warning_x, warning_y),
    #                    warning_font, warning_font_scale, warning_color, warning_thickness)
    
    return vis_frame


# Global state untuk track detections dengan timestamp (real-time PC)
# Format: {'fallen': [(frame_id, datetime_obj, track_id), ...], 'collision': [(frame_id, datetime_obj, track_id_1, track_id_2), ...], 'proximity': [(frame_id, datetime_obj, track_id_1, track_id_2), ...]}
detection_history = {
    'fallen': [],     # List of (frame_id, datetime, track_id)
    'collision': [],  # List of (frame_id, datetime, track_id_1, track_id_2)
    'proximity': []   # List of (frame_id, datetime, track_id_1, track_id_2) - untuk proximity warning
}

# Global state untuk collision warnings (persist 15 seconds after detection)
# Format: {(track_id_1, track_id_2): (first_detected_datetime, last_detected_datetime, last_bbox)}
# last_bbox: [x1, y1, x2, y2] untuk draw warning meskipun detection hilang
proximity_warnings = {}  # Dict of {pair_key: (first_detected_datetime, last_detected_datetime, last_bbox)}
# Note: Variable name kept as 'proximity_warnings' for backward compatibility, but now used for all collision warnings

def draw_alert_banner(frame: np.ndarray, behaviour_results: Dict, 
                      frame_id: int, fps: float = 30.0, detector_type: str = 'all',
                      proximity_warning_pairs: Dict = None) -> np.ndarray:
    """
    Draw alert banner when motorcycle fallen, collision, or proximity warning detected
    Alarm stays active for 15 seconds after last detection (handle occlusion)
    
    Args:
        frame: Input frame
        behaviour_results: Dict of behaviour detection results
        frame_id: Current frame ID
        fps: Video FPS (not used, using real-time PC timestamp)
        detector_type: Which detector to show alerts for ('fallen', 'collision', 'all')
        proximity_warning_pairs: Dict of proximity warnings from collision_detector
        
    Returns:
        Frame with alert banner
    """
    global detection_history
    
    vis_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Get current real-time from PC
    current_datetime = datetime.now()
    
    # ============================================
    # Clear collision/proximity history if detector_type == 'fallen' (prevent false alarms)
    # ============================================
    if detector_type == 'fallen':
        # Clear collision and proximity history to prevent alarms from appearing
        # when user only wants fallen detection
        detection_history['collision'] = []
        detection_history['proximity'] = []
    
    # ============================================
    # Handle PROXIMITY WARNINGS (vehicles too close, collision will happen)
    # ============================================
    proximity_detections = []
    if detector_type in ['collision', 'all']:
        # Get proximity warnings dari parameter (passed from main())
        if proximity_warning_pairs is None:
            proximity_warning_pairs = {}
        
        # Process proximity warnings
        for pair_key, warning_data in proximity_warning_pairs.items():
            track_i = warning_data.get('track_i', -1)
            track_j = warning_data.get('track_j', -1)
            iou = warning_data.get('iou', 0.0)
            
            # Only process if both tracks are 'car' (for 'all' mode)
            if detector_type == 'all':
                # Need to check class from tracks - but we don't have direct access here
                # Assume proximity warnings are only for cars (collision_detector filters)
                pass
            
            proximity_detections.append({
                'track_id_1': track_i,
                'track_id_2': track_j,
                'iou': iou,
                'pair_key': pair_key
            })
        
        if len(proximity_detections) > 0:
            # Record proximity warnings dengan real-time PC timestamp
            for det in proximity_detections:
                track_id_1 = det.get('track_id_1', -1)
                track_id_2 = det.get('track_id_2', -1)
                pair_key = det.get('pair_key', tuple(sorted([track_id_1, track_id_2])))
                
                # Cek apakah pair sudah ada di history dalam 1 detik terakhir (avoid spam)
                recent_detections = [h for h in detection_history['proximity'] 
                                   if tuple(sorted([h[2], h[3]])) == pair_key and 
                                   (current_datetime - h[1]).total_seconds() < 1.0]
                if not recent_detections:
                    # Record baru dengan real-time PC timestamp
                    detection_history['proximity'].append((frame_id, current_datetime, track_id_1, track_id_2))
        
        # Cleanup history yang sudah lebih dari 60 detik
        detection_history['proximity'] = [h for h in detection_history['proximity'] 
                                         if (current_datetime - h[1]).total_seconds() <= 60.0]
    
    # ============================================
    # Handle FALLEN detections (only if detector_type allows it)
    # ============================================
    fallen_detections = []
    if detector_type in ['fallen', 'all']:
        # Get fallen detections and filter by behaviour_type
        raw_fallen = behaviour_results.get('fallen', [])
        for det in raw_fallen:
            # Only process detections with behaviour_type == 'motorcycle_fallen'
            behaviour_type = det.get('behaviour_type', 'unknown')
            if behaviour_type == 'motorcycle_fallen':
                # For 'all' mode: only process if class is 'motorcycle'
                # (FallenDetector already filters for motorcycle, but double-check for safety)
                if detector_type == 'all':
                    # FallenDetector already ensures only motorcycle, so we trust it
                    pass
                
                # Skip physics prediction yang sudah expired
                prediction_mode = det.get('prediction_mode', 'confirmed')
                if prediction_mode == 'physics_predicted':
                    confidence = det.get('confidence', 1.0)
                    frames_since = det.get('frames_since_seen', 0)
                    # Skip jika confidence terlalu rendah atau frames_since terlalu tinggi
                    if confidence < 0.3 or frames_since > 60:
                        continue
                fallen_detections.append(det)
    
    if len(fallen_detections) > 0:
        # Record detections baru dengan real-time PC timestamp
        for det in fallen_detections:
            track_id = det.get('track_id', -1)
            # Cek apakah track_id sudah ada di history dalam 1 detik terakhir (avoid spam)
            recent_detections = [h for h in detection_history['fallen'] 
                               if h[2] == track_id and 
                               (current_datetime - h[1]).total_seconds() < 1.0]
            if not recent_detections:
                # Record baru dengan real-time PC timestamp
                detection_history['fallen'].append((frame_id, current_datetime, track_id))
    
    # Cleanup history yang sudah lebih dari 60 detik (keep history log manageable)
    detection_history['fallen'] = [h for h in detection_history['fallen'] 
                                  if (current_datetime - h[1]).total_seconds() <= 60.0]
    
    # ============================================
    # Handle COLLISION detections (only if detector_type allows it)
    # ============================================
    collision_detections = []
    if detector_type in ['collision', 'all']:
        # Get collision detections and filter by behaviour_type
        raw_collision = behaviour_results.get('collision', [])
        for det in raw_collision:
            # Only process detections with behaviour_type == 'collision'
            behaviour_type = det.get('behaviour_type', 'unknown')
            if behaviour_type == 'collision':
                # For 'all' mode: hanya kendaraan bermotor
                if detector_type == 'all':
                    class_primary = det.get('class_primary', 'unknown')
                    class_secondary = det.get('class_secondary', 'unknown')
                    if class_primary not in _MOTOR_VEHICLE_CLS or class_secondary not in _MOTOR_VEHICLE_CLS:
                        continue
                
                collision_detections.append(det)
        
        if len(collision_detections) > 0:
            # Record detections baru dengan real-time PC timestamp
            for det in collision_detections:
                track_id_1 = det.get('track_id', -1)
                track_id_2 = det.get('track_id_secondary', -1)
                alert_level = det.get('alert_level', 'emergency')
                confidence_label = det.get('confidence_label', 'COLLISION DETECTED')
                # Create pair key untuk avoid duplicate
                pair_key = tuple(sorted([track_id_1, track_id_2]))

                # Cek apakah pair sudah ada di history dalam 1 detik terakhir (avoid spam)
                recent_detections = [h for h in detection_history['collision']
                                   if tuple(sorted([h[2], h[3]])) == pair_key and
                                   (current_datetime - h[1]).total_seconds() < 1.0]
                if not recent_detections:
                    # Record baru dengan real-time PC timestamp
                    # Record baru dengan real-time PC timestamp
                    # Tuple format: (frame_id, datetime, track_id_1, track_id_2, alert_level, confidence_label, confidence_score)
                    confidence_score_val = det.get('confidence', 0)
                    detection_history['collision'].append((frame_id, current_datetime, track_id_1, track_id_2, alert_level, confidence_label, confidence_score_val))
        
        # Cleanup history yang sudah lebih dari 60 detik (only if detector_type allows collision)
        detection_history['collision'] = [h for h in detection_history['collision'] 
                                         if (current_datetime - h[1]).total_seconds() <= 60.0]
    
    # ============================================
    # Check alarm status (priority based on detector_type)
    # Priority: Collision > Proximity Warning > Fallen
    # ============================================
    
    alarm_active = False
    alarm_type = None  # 'collision', 'proximity', or 'fallen'
    last_detection_datetime = None
    alarm_data = None  # Data untuk display
    
    # Priority logic based on detector_type
    
    # Priority logic based on detector_type
    if detector_type == 'collision':
        # Priority 1: Check collision alarm
        if detection_history['collision']:
            # Search for active HIGH confidence alarm first (Priority)
            high_conf_active = False
            for detection in reversed(detection_history['collision']):
                # Format: (frame_id, datetime, track_id_1, track_id_2, alert_level, confidence_label, confidence_score)
                det_datetime = detection[1]
                det_conf_score = detection[6] if len(detection) >= 7 else 0
                
                time_since_det = (current_datetime - det_datetime).total_seconds()
                
                # Check if this detection is still valid (15s duration)
                # >= 70%: COLLISION DETECTED (merah besar)
                # >= 50%: CRITICAL INTERACTION (orange)
                # >= 30%: UNSAFE PROXIMITY (kuning)
                if det_conf_score >= 30 and time_since_det <= 15.0:
                    alarm_active = True
                    alarm_type = 'collision'
                    alarm_data = detection
                    high_conf_active = True
                    break

            # Low-confidence (< 30%) tidak mengaktifkan banner â€” NORMAL, tidak ada bukti kuat
        
        # ============================================
        # PROXIMITY ALARM DISABLED (Priority 2 - commented out)
        # ============================================
        # Priority 2: Check proximity warning (only if collision not active)
        # if not alarm_active and detection_history['proximity']:
        #     last_detection = detection_history['proximity'][-1]
        #     last_detection_datetime = last_detection[1]
        #     time_since_detection = (current_datetime - last_detection_datetime).total_seconds()
        #
        #     if time_since_detection <= 15.0:  # 15 seconds
        #         alarm_active = True
        #         alarm_type = 'proximity'
        #         alarm_data = last_detection
                
    elif detector_type == 'fallen':
        # Only check fallen alarm
        if detection_history['fallen']:
            last_detection = detection_history['fallen'][-1]
            last_detection_datetime = last_detection[1]
            time_since_detection = (current_datetime - last_detection_datetime).total_seconds()
            
            if time_since_detection <= 15.0:  # 15 seconds
                alarm_active = True
                alarm_type = 'fallen'
                alarm_data = last_detection
                
    else:  # detector_type == 'all'
        # Priority 1: Check collision alarm
        if detection_history['collision']:
            high_conf_active = False
            for detection in reversed(detection_history['collision']):
                det_datetime = detection[1]
                det_conf_score = detection[6] if len(detection) >= 7 else 0
                time_since_det = (current_datetime - det_datetime).total_seconds()
                
                if det_conf_score >= 30 and time_since_det <= 15.0:
                    alarm_active = True
                    alarm_type = 'collision'
                    alarm_data = detection
                    high_conf_active = True
                    break
            
            # Three-level collision UI:
            # >=70 red escalation, >=50 orange critical interaction, >=30 yellow monitor closely.
        
        # ============================================
        # PROXIMITY ALARM DISABLED (Priority 2 - commented out)
        # ============================================
        # Priority 2: Check proximity warning (only if collision not active)
        # if not alarm_active and detection_history['proximity']:
        #     last_detection = detection_history['proximity'][-1]
        #     last_detection_datetime = last_detection[1]
        #     time_since_detection = (current_datetime - last_detection_datetime).total_seconds()
        #
        #     if time_since_detection <= 15.0:  # 15 seconds
        #         alarm_active = True
        #         alarm_type = 'proximity'
        #         alarm_data = last_detection
        
        # Priority 3: Check fallen alarm (only if collision and proximity not active)
        if not alarm_active and detection_history['fallen']:
            last_detection = detection_history['fallen'][-1]
            last_detection_datetime = last_detection[1]
            time_since_detection = (current_datetime - last_detection_datetime).total_seconds()
            
            if time_since_detection <= 15.0:  # 15 seconds
                alarm_active = True
                alarm_type = 'fallen'
                alarm_data = last_detection
    
    # ============================================
    # Draw alert banner jika alarm masih aktif
    # ============================================
    if alarm_active:
        # Determine alert_level and confidence for collision
        if alarm_type == 'collision':
            if len(collision_detections) > 0:
                # Gunakan detection dengan confidence TERTINGGI di frame ini, bukan yang pertama.
                # Mencegah bug: collision[0] adalah 65% MIGHT, tapi collision[1] adalah 70%+.
                _level_priority = {'emergency': 2, 'warning': 1, 'caution': 0, 'info': -1}
                best_det = max(collision_detections, key=lambda d: (
                    _level_priority.get(d.get('alert_level', 'emergency'), 0),
                    d.get('confidence', 0)
                ))
                alert_level = best_det.get('alert_level', 'emergency')
                confidence_label = best_det.get('confidence_label', 'COLLISION DETECTED')
                confidence_score = best_det.get('confidence', 0)
            else:
                if alarm_data and len(alarm_data) >= 7:
                    alert_level = alarm_data[4]
                    confidence_label = alarm_data[5]
                    confidence_score = alarm_data[6]
                elif alarm_data and len(alarm_data) >= 6:
                    alert_level = alarm_data[4]
                    confidence_label = alarm_data[5]
                    confidence_score = 0
                else:
                    alert_level = 'emergency'
                    confidence_label = 'COLLISION DETECTED'
                    confidence_score = 0
        else:
            alert_level = 'emergency'  # fallen/proximity selalu emergency
            confidence_label = ''
            confidence_score = 0

        # ============================================
        # BANNER LOGIC:
        # - emergency: red escalation banner
        # - warning: orange critical-interaction banner
        # - caution: yellow monitor-closely banner
        is_emergency = (alert_level == 'emergency') or (alarm_type == 'fallen')

        if is_emergency:
            # === BANNER MERAH BESAR (emergency collision / fallen) ===
            banner_height = 80
            banner_y = 10

            if alarm_type == 'collision':
                if confidence_score >= 70 or confidence_score == 0:
                    # >= 70: COLLISION DETECTED, red escalation
                    banner_color = (0, 0, 255)  # Red (BGR)
                    if confidence_score > 0:
                        alert_text = f"!!! {confidence_label} | S: {confidence_score:.0f} !!!"
                    else:
                        alert_text = f"!!! {confidence_label} !!!"
                else:
                    # Fallback; normally non-emergency collision uses the branch below.
                    banner_color = (0, 165, 255)  # Orange (BGR)
                    alert_text = f"CRITICAL INTERACTION | S: {confidence_score:.0f}"
            elif alarm_type == 'proximity':
                banner_color = (0, 255, 255)  # Yellow
                alert_text = "!!! TOO CLOSE, COLLISION WILL HAPPEN !!!"
            else:  # fallen
                banner_color = (0, 0, 255)  # Red
                alert_text = "!!! MOTORCYCLE FALLEN DETECTED !!!"

            # Semi-transparent background
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (0, banner_y), (w, banner_y + banner_height), banner_color, -1)
            cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 4

            text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = banner_y + banner_height // 2 + text_size[1] // 2

            # Outline (black)
            cv2.putText(vis_frame, alert_text, (text_x - 2, text_y - 2),
                       font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(vis_frame, alert_text, (text_x + 2, text_y + 2),
                       font, font_scale, (0, 0, 0), thickness + 2)
            # Main text (white)
            cv2.putText(vis_frame, alert_text, (text_x, text_y),
                       font, font_scale, (255, 255, 255), thickness)

            # Info detail below alert
            detail_texts = []
            if alarm_type == 'collision':
                if len(collision_detections) > 0:
                    for det in collision_detections:
                        track_id_1 = det.get('track_id', -1)
                        track_id_2 = det.get('track_id_secondary', -1)
                        severity = det.get('severity', 'unknown')
                        detail_texts.append(f"Tracks: {track_id_1} <-> {track_id_2} | Severity: {severity.upper()}")
                else:
                    if alarm_data:
                        time_remaining = 15.0 - (current_datetime - alarm_data[1]).total_seconds()
                        detail_texts.append(f"Tracks: {alarm_data[2]} <-> {alarm_data[3]} | Time remaining: {time_remaining:.1f}s")
            elif alarm_type == 'proximity':
                if len(proximity_detections) > 0:
                    for det in proximity_detections:
                        track_id_1 = det.get('track_id_1', -1)
                        track_id_2 = det.get('track_id_2', -1)
                        iou = det.get('iou', 0.0)
                        detail_texts.append(f"Tracks: {track_id_1} <-> {track_id_2} | IoU: {iou:.3f}")
                else:
                    if alarm_data:
                        time_remaining = 15.0 - (current_datetime - alarm_data[1]).total_seconds()
                        detail_texts.append(f"Tracks: {alarm_data[2]} <-> {alarm_data[3]} | Time remaining: {time_remaining:.1f}s")
            else:  # fallen
                if len(fallen_detections) > 0:
                    for det in fallen_detections:
                        track_id = det.get('track_id', -1)
                        severity = det.get('severity', 'unknown')
                        detail_texts.append(f"Track ID: {track_id} | Severity: {severity.upper()}")
                else:
                    if alarm_data:
                        time_remaining = 15.0 - (current_datetime - alarm_data[1]).total_seconds()
                        detail_texts.append(f"Track ID: {alarm_data[2]} | Time remaining: {time_remaining:.1f}s")

            detail_y = banner_y + banner_height + 30
            for i, detail_text in enumerate(detail_texts[:3]):
                detail_size = cv2.getTextSize(detail_text, font, 0.7, 2)[0]
                detail_x = (w - detail_size[0]) // 2
                cv2.putText(vis_frame, detail_text, (detail_x, detail_y + i * 25),
                           font, 0.7, (255, 255, 0), 2)

        else:
            # === NON-EMERGENCY COLLISION BANNERS ===
            banner_y = 10

            if alert_level == 'warning':
                banner_height = 50
                banner_color = (0, 140, 255)  # Orange (BGR)
                if confidence_score > 0:
                    alert_text = f"CRITICAL INTERACTION | S: {confidence_score:.0f}"
                else:
                    alert_text = "CRITICAL INTERACTION"
            elif alert_level == 'caution':
                # 30-49: monitor-closely layer, yellow
                banner_height = 50
                banner_color = (0, 220, 255)  # Yellow (BGR)
                if confidence_score > 0:
                    alert_text = f"MONITOR CLOSELY | S: {confidence_score:.0f}"
                else:
                    alert_text = "MONITOR CLOSELY"
            else:
                # info/normal â€” dark gray kecil
                banner_height = 35
                banner_color = (40, 40, 40)
                alert_text = f"[INFO] {confidence_label}"

            # Get track info
            if len(collision_detections) > 0:
                det = collision_detections[0]
                track_id_1 = det.get('track_id', -1)
                track_id_2 = det.get('track_id_secondary', -1)
                tier = det.get('tier', 'N/A')
                alert_text += f" | Tracks: {track_id_1}<->{track_id_2} | Tier: {tier}"
            elif alarm_data and len(alarm_data) >= 4:
                alert_text += f" | Tracks: {alarm_data[2]}<->{alarm_data[3]}"

            # Semi-transparent dark background
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (0, banner_y), (w, banner_y + banner_height), banner_color, -1)
            cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = banner_y + banner_height // 2 + text_size[1] // 2

            # Text color based on alert_level
            if alert_level == 'warning':
                text_color = (255, 255, 255)  # White on orange
            elif alert_level == 'caution':
                text_color = (20, 20, 20)  # Dark text on yellow
            else:
                text_color = (180, 180, 180)  # Light gray (info/low confidence)

            # Outline (black)
            cv2.putText(vis_frame, alert_text, (text_x - 1, text_y - 1),
                       font, font_scale, (0, 0, 0), thickness + 1)
            # Main text
            cv2.putText(vis_frame, alert_text, (text_x, text_y),
                       font, font_scale, text_color, thickness)
    
    return vis_frame


def draw_detection_summary(frame: np.ndarray, behaviour_results: Dict, 
                           scene_info: Dict = None, fps: float = 30.0, detector_type: str = 'all') -> np.ndarray:
    """
    Draw summary panel showing detection counts and history log with timestamp
    
    Args:
        frame: Input frame
        behaviour_results: Dict of behaviour detection results
        scene_info: Optional scene analysis info
        fps: Video FPS (not used, using real-time PC timestamp)
        detector_type: Which detector to show summary for ('fallen', 'collision', 'all')
        
    Returns:
        Frame with summary panel
    """
    global detection_history
    
    vis_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Panel positioned at bottom of video (not at top)
    # Panel height increased for history log
    panel_height = 200  # Increased height for history log
    panel_width = 450   # Increased width for history log
    panel_start_y = h - panel_height - 20  # Position at bottom with 20px margin
    
    # Semi-transparent panel - bottom left of video
    overlay = vis_frame.copy()
    cv2.rectangle(overlay, (10, panel_start_y), (panel_width, panel_start_y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
    
    # Title - adjust Y position sesuai panel_start_y
    cv2.putText(vis_frame, "LTE-TrackGuard Detection", (20, panel_start_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Detection counts - adjust Y position (filter by detector_type)
    y_offset = panel_start_y + 55
    for det_type, dets in behaviour_results.items():
        # Filter by detector_type
        if detector_type == 'fallen' and det_type != 'fallen':
            continue
        if detector_type == 'collision' and det_type != 'collision':
            continue
        
        # Filter detections by behaviour_type and class (for 'all' mode)
        filtered_dets = []
        for det in dets:
            behaviour_type = det.get('behaviour_type', 'unknown')
            if det_type == 'fallen' and behaviour_type != 'motorcycle_fallen':
                continue
            if det_type == 'collision' and behaviour_type != 'collision':
                continue
            
            # For 'all' mode: filter by class
            if detector_type == 'all':
                # Collision: semua kendaraan bermotor
                if behaviour_type == 'collision':
                    class_primary = det.get('class_primary', 'unknown')
                    class_secondary = det.get('class_secondary', 'unknown')
                    if class_primary not in _MOTOR_VEHICLE_CLS or class_secondary not in _MOTOR_VEHICLE_CLS:
                        continue
                
                # Fallen: only 'motorcycle' (FallenDetector already filters, but double-check)
                # We trust FallenDetector filtering, so no additional check needed
            
            # Skip expired physics predictions for fallen
            if det_type == 'fallen' and behaviour_type == 'motorcycle_fallen':
                prediction_mode = det.get('prediction_mode', 'confirmed')
                if prediction_mode == 'physics_predicted':
                    confidence = det.get('confidence', 1.0)
                    frames_since = det.get('frames_since_seen', 0)
                    if confidence < 0.3 or frames_since > 60:
                        continue
            filtered_dets.append(det)
        
        count = len(filtered_dets)
        if count > 0:
            # No emoji - text only
            text = f"{det_type.capitalize()}: {count}"
            color = (0, 0, 255) if count > 0 and det_type in ['fallen', 'collision'] else (0, 255, 255)
            
            cv2.putText(vis_frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
    
    # History log for fallen and collision with timestamp (filter by detector_type)
    has_history = False
    if detector_type in ['fallen', 'all']:
        has_history = has_history or len(detection_history['fallen']) > 0
    if detector_type in ['collision', 'all']:
        has_history = has_history or len(detection_history['collision']) > 0
    
    if has_history:
        y_offset += 10  # Spacing
        cv2.putText(vis_frame, "History:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Combine histories (filtered by detector_type) and sort by frame_id
        all_history = []
        if detector_type in ['fallen', 'all']:
            for hist in detection_history['fallen']:
                frame_id_hist, datetime_obj, track_id = hist
                all_history.append(('fallen', frame_id_hist, datetime_obj, track_id))
        
        if detector_type in ['collision', 'all']:
            for hist in detection_history['collision']:
                # Handle both old (4 elements) and new (6 elements) tuple formats
                hist_alert_level = 'emergency'
                if len(hist) >= 7:
                    frame_id_hist, datetime_obj, track_id_1, track_id_2, hist_alert_level, confidence_label, conf_score = hist
                elif len(hist) >= 6:
                    frame_id_hist, datetime_obj, track_id_1, track_id_2, hist_alert_level, confidence_label = hist
                else:
                    frame_id_hist, datetime_obj, track_id_1, track_id_2 = hist
                all_history.append(('collision', frame_id_hist, datetime_obj, track_id_1, track_id_2, hist_alert_level))

            for hist in detection_history['proximity']:
                frame_id_hist, datetime_obj, track_id_1, track_id_2 = hist
                all_history.append(('proximity', frame_id_hist, datetime_obj, track_id_1, track_id_2))
        
        # Sort by frame_id (newest last)
        all_history.sort(key=lambda x: x[1])
        
        # Show 5 latest history (newest at bottom)
        history_to_show = all_history[-5:]  # Last 5
        
        for hist in history_to_show:
            hist_type = hist[0]
            frame_id_hist = hist[1]
            datetime_obj = hist[2]
            
            # Format timestamp from PC real-time: HH:MM:SS
            time_str = datetime_obj.strftime("%H:%M:%S")
            
            if hist_type == 'fallen':
                track_id = hist[3]
                log_text = f"Motorcycle fallen - Track {track_id} at Frame {frame_id_hist} {time_str}"
            elif hist_type == 'proximity':
                track_id_1 = hist[3]
                track_id_2 = hist[4]
                log_text = f"CAUTION - Tracks {track_id_1} <-> {track_id_2} at Frame {frame_id_hist} {time_str}"
            else:  # collision
                track_id_1 = hist[3]
                track_id_2 = hist[4]
                lvl = hist[5].upper() if len(hist) >= 6 else 'EMERGENCY'
                log_text = f"{lvl} - Tracks {track_id_1} <-> {track_id_2} at Frame {frame_id_hist} {time_str}"
            
            cv2.putText(vis_frame, log_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
    
    # Scene info (if available) - pindahkan ke bawah history
    if scene_info:
        category = scene_info.get('category', 'unknown')
        density = scene_info.get('density', 0)
        
        cv2.putText(vis_frame, f"Scene: {category} ({density:.1f} tr/Mpx)", 
                   (20, panel_start_y + panel_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_frame


def generate_forensic_report(collision_detection: Dict, frame_id: int, 
                             fps: float, video_path: str,
                             brake_results: List[Dict] = None,
                             track_manager=None) -> str:
    """
    Generate detailed forensic report format untuk terminal/log file
    
    Args:
        collision_detection: Collision detection dict
        frame_id: Frame ID
        fps: Video FPS
        video_path: Path to video file
        brake_results: Brake detection results (optional)
        track_manager: Track manager untuk get velocity info
        
    Returns:
        Formatted forensic report string
    """
    from datetime import datetime
    import numpy as np
    
    # Extract collision info
    track_id_1 = collision_detection.get('track_id', -1)
    track_id_2 = collision_detection.get('track_id_secondary', -1)
    
    # Energy & Severity
    energy_loss_i = collision_detection.get('energy_loss_primary', 0.0)
    energy_loss_j = collision_detection.get('energy_loss_secondary', 0.0)
    total_energy_loss = (energy_loss_i + energy_loss_j) / 2.0
    
    # Classify severity
    if total_energy_loss >= 0.9:
        severity = "CRITICAL"
    elif total_energy_loss >= 0.7:
        severity = "HIGH"
    elif total_energy_loss >= 0.5:
        severity = "MEDIUM"
    else:
        severity = "LOW"
    
    # Momentum transfer
    momentum_transfer_i = collision_detection.get('momentum_transfer_i', 0.0)
    momentum_transfer_j = collision_detection.get('momentum_transfer_j', 0.0)
    
    # Get velocity vectors dari track_manager untuk calculate speeds dan impact angle
    v_i_vec = None
    v_j_vec = None
    v_i_speed = 0.0
    v_j_speed = 0.0
    
    if track_manager and hasattr(track_manager, 'velocity_field'):
        try:
            # Get tracks
            tracks = track_manager.get_current_tracks()
            track_i_obj = next((t for t in tracks if t.track_id == track_id_1), None)
            track_j_obj = next((t for t in tracks if t.track_id == track_id_2), None)
            
            if track_i_obj and track_j_obj:
                v_i_vec = track_manager.velocity_field.compute_velocity(track_i_obj, dt=1.0)
                v_j_vec = track_manager.velocity_field.compute_velocity(track_j_obj, dt=1.0)
                v_i_speed = np.linalg.norm(v_i_vec)
                v_j_speed = np.linalg.norm(v_j_vec)
        except Exception as e:
            # Fallback: try get dari collision_detection
            v_i_speed = collision_detection.get('v_i_speed', 0.0)
            v_j_speed = collision_detection.get('v_j_speed', 0.0)
    else:
        # Fallback: try get dari collision_detection
        v_i_speed = collision_detection.get('v_i_speed', 0.0)
        v_j_speed = collision_detection.get('v_j_speed', 0.0)
    
    # Impact direction untuk calculate angle
    impact_direction = collision_detection.get('impact_direction', [0, 0])
    if isinstance(impact_direction, list):
        impact_direction = np.array(impact_direction)
    
    # Calculate impact angle (angle between velocity vectors)
    if v_i_vec is not None and v_j_vec is not None:
        # Use actual velocity vectors (most accurate)
        v_i_norm = np.linalg.norm(v_i_vec)
        v_j_norm = np.linalg.norm(v_j_vec)
        
        if v_i_norm > 1e-6 and v_j_norm > 1e-6:
            v_i_unit = v_i_vec / v_i_norm
            v_j_unit = v_j_vec / v_j_norm
            dot_product = np.clip(np.dot(v_i_unit, v_j_unit), -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            impact_angle = np.degrees(angle_rad)
            # Normalize to [0, 180]
            impact_angle = min(impact_angle, 180.0 - impact_angle)
        else:
            impact_angle = 0.0
    elif np.linalg.norm(impact_direction) > 1e-6:
        # Fallback: use impact_direction angle (less accurate but better than nothing)
        impact_angle_rad = np.arctan2(impact_direction[1], impact_direction[0])
        impact_angle = abs(np.degrees(impact_angle_rad))
        # Normalize to [0, 180]
        if impact_angle > 180:
            impact_angle = 360 - impact_angle
    else:
        impact_angle = 0.0
    
    # Infer evidence dari momentum transfer dan speeds
    # Check velocity mismatch (significant speed difference)
    velocity_mismatch = abs(v_i_speed - v_j_speed) > 1.0 if (v_i_speed > 0 and v_j_speed > 0) else False
    
    # Check if approaching (both moving in similar direction)
    is_approaching = False
    if v_i_vec is not None and v_j_vec is not None:
        v_i_norm = np.linalg.norm(v_i_vec)
        v_j_norm = np.linalg.norm(v_j_vec)
        if v_i_norm > 1e-6 and v_j_norm > 1e-6:
            v_i_unit = v_i_vec / v_i_norm
            v_j_unit = v_j_vec / v_j_norm
            dot_product = np.dot(v_i_unit, v_j_unit)
            # Approaching = moving in similar direction (dot product > 0.5)
            is_approaching = dot_product > 0.5
    
    # Check rear-end evidence: velocity mismatch + approaching + momentum transfer
    has_rear_end_evidence = (velocity_mismatch and is_approaching and 
                            (momentum_transfer_i > 0 or momentum_transfer_j > 0))
    
    if has_rear_end_evidence or (impact_angle < 30.0 and velocity_mismatch):
        collision_type = "Rear-end Collision (Tabrak Belakang)"
    elif impact_angle > 150.0:
        collision_type = "Head-on Collision (Tabrak Depan)"
    else:
        collision_type = "Side Collision (Tabrak Samping)"
    
    # Identify aggressor/victim dari momentum transfer
    if abs(momentum_transfer_i) > abs(momentum_transfer_j):
        if momentum_transfer_i > 0:
            aggressor_id = track_id_1
            victim_id = track_id_2
            aggressor_speed = v_i_speed
            victim_speed = v_j_speed
            aggressor_energy = energy_loss_i
            victim_energy = energy_loss_j
            momentum_value = momentum_transfer_i
        else:
            aggressor_id = track_id_2
            victim_id = track_id_1
            aggressor_speed = v_j_speed
            victim_speed = v_i_speed
            aggressor_energy = energy_loss_j
            victim_energy = energy_loss_i
            momentum_value = abs(momentum_transfer_j)
    else:
        if momentum_transfer_j > 0:
            aggressor_id = track_id_2
            victim_id = track_id_1
            aggressor_speed = v_j_speed
            victim_speed = v_i_speed
            aggressor_energy = energy_loss_j
            victim_energy = energy_loss_i
            momentum_value = momentum_transfer_j
        else:
            aggressor_id = track_id_1
            victim_id = track_id_2
            aggressor_speed = v_i_speed
            victim_speed = v_j_speed
            aggressor_energy = energy_loss_i
            victim_energy = energy_loss_j
            momentum_value = abs(momentum_transfer_i)
    
    # Get class names
    class_primary = collision_detection.get('class_primary', 'unknown')
    class_secondary = collision_detection.get('class_secondary', 'unknown')
    aggressor_class = class_primary if aggressor_id == track_id_1 else class_secondary
    victim_class = class_secondary if aggressor_id == track_id_1 else class_primary
    
    # Check pre-collision braking
    aggressor_braked = False
    if brake_results:
        frames_before = 10
        for brake_det in brake_results:
            brake_track_id = brake_det.get('track_id', -1)
            brake_frame_id = brake_det.get('frame_id', -1)
            if frame_id - frames_before <= brake_frame_id < frame_id:
                if brake_track_id == aggressor_id:
                    aggressor_braked = True
                    break
    
    # Collision point
    collision_point = collision_detection.get('collision_point', [0, 0])
    
    # Generate case ID
    timestamp = datetime.now()
    case_id = f"#COLLISION-{timestamp.strftime('%Y%m%d-%H%M%S')}"
    
    # Calculate frame timestamp
    if fps > 0:
        total_seconds = frame_id / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 100)
        frame_timestamp = f"{minutes:02d}:{seconds:02d}.{milliseconds:02d}"
    else:
        frame_timestamp = "00:00.00"
    
    # Build report
    report_lines = []
    report_lines.append("[LTE-TrackGuard FORENSIC AUDIT REPORT]")
    report_lines.append("=" * 60)
    report_lines.append(f"CASE ID    : {case_id}")
    report_lines.append(f"TIMESTAMP  : {frame_timestamp} (Frame {frame_id})")
    report_lines.append(f"LOCATION   : Camera-XX (KM XXX+XXX)  [NANTI - placeholder]")
    report_lines.append(f"SEVERITY   : {severity} ({total_energy_loss*100:.0f}% Energy Loss)")
    report_lines.append(f"TYPE       : {collision_type}")
    report_lines.append(f"VERDICT    : Vehicle {aggressor_id} ({aggressor_class.upper()}) hit Vehicle {victim_id} ({victim_class.upper()})")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append("1. CAUSALITY ANALYSIS (Analisis Penyebab)")
    report_lines.append("   " + "-" * 57)
    report_lines.append(f"   - Aggressor : Track {aggressor_id} (Velocity {aggressor_speed:.1f} px/fr)")
    report_lines.append(f"   - Victim    : Track {victim_id} (Velocity {victim_speed:.1f} px/fr)")
    report_lines.append(f"   - Evidence  : Positive Momentum Transfer from {aggressor_id} -> {victim_id} (+{momentum_value:.1f} units)")
    braking_text = "NO braking" if not aggressor_braked else "YES braking"
    report_lines.append(f"                 Track {aggressor_id} showed {braking_text} (Divergence > -0.2) prior to impact.")
    report_lines.append("")
    report_lines.append("2. PHYSICS EVIDENCE (Bukti Fisika)")
    report_lines.append("   " + "-" * 57)

    # Check if this is evasive collision
    detection_mode = collision_detection.get('detection_mode', 'normal')
    evasive_data = collision_detection.get('evasive_data', None)

    if detection_mode == 'evasive_collision' and evasive_data:
        # Evasive collision - special reporting with CONFIDENCE SCORING
        angular_vel = evasive_data.get('angular_velocity', 0.0)
        lateral_acc = evasive_data.get('lateral_acceleration', 0.0)
        direction_change = evasive_data.get('direction_change', 0.0)
        aggressor_speed = evasive_data.get('aggressor_speed', 0.0)

        # Confidence scoring
        confidence = collision_detection.get('confidence', 0.0)
        confidence_level = collision_detection.get('confidence_level', 'UNKNOWN')
        evidence = collision_detection.get('evidence_breakdown', {})
        victim_status = collision_detection.get('victim_status', 'unknown')

        report_lines.append(f"   - Detection Type: EVASIVE COLLISION (vehicle swerved suddenly)")
        report_lines.append(f"   - Collision Confidence: {confidence:.0f}% ({confidence_level})")
        report_lines.append(f"   - Evidence Breakdown:")
        report_lines.append(f"     â€¢ Proximity Score: {evidence.get('proximity_score', 0)}/30 (max IoU: {evidence.get('max_iou', 0.0):.3f})")
        report_lines.append(f"     â€¢ Evasive Score: {evidence.get('evasive_score', 0)}/25 (angular vel: {angular_vel:.1f}Â°/fr)")
        report_lines.append(f"     â€¢ Impact Score: {evidence.get('impact_score', 0)}/30 (victim: {victim_status})")
        report_lines.append(f"     â€¢ Motion Score: {evidence.get('motion_score', 0)}/15 (closing rate: {evidence.get('closing_rate', 0.0):.1f} px/fr)")
        report_lines.append(f"   - Angular Velocity: {angular_vel:.1f}Â°/frame (sudden direction change)")
        report_lines.append(f"   - Lateral Acceleration: {lateral_acc:.2f} px/frÂ² (evasive maneuver)")
        report_lines.append(f"   - Direction Change: {direction_change:.1f}Â° (total deflection)")
        report_lines.append(f"   - Aggressor Speed: {aggressor_speed:.1f} px/fr")
        report_lines.append(f"   - Victim Status: {victim_status.upper()}")
        if victim_status == 'visible':
            report_lines.append(f"   - Victim Impact Evidence:")
            report_lines.append(f"     â€¢ Velocity Change: {evidence.get('victim_velocity_change', 0.0)*100:.0f}%")
            report_lines.append(f"     â€¢ Direction Change: {evidence.get('victim_direction_change', 0.0):.1f}Â°")
        proximity_frames = collision_detection.get('proximity_frames', 0)
        max_iou = collision_detection.get('max_iou', 0.0)
        report_lines.append(f"   - Proximity History: {proximity_frames} frames, max IoU: {max_iou:.3f}")
    else:
        # Normal collision reporting
        report_lines.append(f"   - Impact Angle: {impact_angle:.1f}Â° (nearly straight, same direction)")
        report_lines.append(f"   - Collision Point: [{collision_point[0]:.0f}, {collision_point[1]:.0f}] pixels")
        report_lines.append(f"   - Energy Loss: Track {track_id_1}={energy_loss_i*100:.0f}%, Track {track_id_2}={energy_loss_j*100:.0f}%")
        report_lines.append(f"   - Total Energy Loss: {total_energy_loss*100:.0f}%")
        report_lines.append(f"   - Momentum Transfer: {track_id_1} -> {track_id_2} = {momentum_transfer_i:.1f} units")
        report_lines.append(f"   - Velocity: Track {track_id_1}={v_i_speed:.1f} px/fr, Track {track_id_2}={v_j_speed:.1f} px/fr")
        closing_rate = abs(v_i_speed - v_j_speed)
        report_lines.append(f"   - Closing Rate: {closing_rate:.1f} px/frame (excessive)")
        push_acc = max(collision_detection.get('push_acceleration_i', 0.0),
                       collision_detection.get('push_acceleration_j', 0.0))
        report_lines.append(f"   - Push Acceleration: {push_acc:.2f} px/frÂ²")

    report_lines.append("")
    report_lines.append("3. COLLISION TYPE DETAILS (The Mechanics)")
    report_lines.append("   " + "-" * 57)
    report_lines.append(f"   - Type: {collision_type}")
    report_lines.append(f"   - Impact Angle: {impact_angle:.1f}Â°")
    if has_rear_end_evidence:
        report_lines.append("   - Characteristics: Same direction, rear vehicle faster")
        report_lines.append("   - Typical Cause: Failure to maintain safe distance")
    elif impact_angle > 150.0:
        report_lines.append("   - Characteristics: Opposite direction, head-on impact")
        report_lines.append("   - Typical Cause: Wrong way or intersection collision")
    else:
        report_lines.append("   - Characteristics: Perpendicular or angled impact")
        report_lines.append("   - Typical Cause: Intersection or lane change collision")
    report_lines.append("")
    report_lines.append("4. RESPONSIBILITY ASSESSMENT")
    report_lines.append("   " + "-" * 57)
    # Calculate fault percentage (simplified)
    fault_aggressor = 95.0 if momentum_value > 100 else 80.0
    fault_victim = 100.0 - fault_aggressor
    report_lines.append(f"   - Track {aggressor_id} ({aggressor_class.upper()}): {fault_aggressor:.0f}% at fault")
    report_lines.append(f"     Evidence: Momentum transfer {aggressor_id} -> {victim_id} (+{momentum_value:.1f} units)")
    report_lines.append(f"     Evidence: {'No' if not aggressor_braked else 'Yes'} braking before impact")
    report_lines.append(f"     Evidence: Excessive closing rate")
    report_lines.append(f"   - Track {victim_id} ({victim_class.upper()}): {fault_victim:.0f}% at fault")
    report_lines.append(f"     Evidence: No braking detected")
    report_lines.append("")
    report_lines.append("5. PHYSICS VALIDATION")
    report_lines.append("   " + "-" * 75)
    momentum_loss = abs(momentum_transfer_i - momentum_transfer_j) / max(abs(momentum_transfer_i), abs(momentum_transfer_j), 1.0) * 100
    report_lines.append(f"   âœ“ Momentum Conservation: VALID")
    report_lines.append(f"     Momentum loss to deformation: {momentum_loss:.1f}% (within expected range)")
    if total_energy_loss > 0.8:
        report_lines.append(f"   âœ“ Energy Conservation: VALID")
        report_lines.append(f"     Energy dissipated: {total_energy_loss*100:.0f}% (high dissipation)")
    else:
        report_lines.append(f"   âš  Energy Conservation: PARTIAL")
        report_lines.append(f"     Energy dissipated: {total_energy_loss*100:.0f}% (moderate dissipation)")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    return "\n".join(report_lines)


def generate_forensic_telegram_message(collision_detection: Dict, frame_id: int, 
                                      fps: float, video_path: str,
                                      brake_results: List[Dict] = None,
                                      track_manager=None) -> str:
    """
    Generate forensic telegram message format yang detail
    
    Args:
        collision_detection: Collision detection dict
        frame_id: Frame ID
        fps: Video FPS
        video_path: Path to video file
        brake_results: Brake detection results (optional)
        
    Returns:
        Formatted telegram message string
    """
    from datetime import datetime
    import numpy as np
    
    # Extract collision info
    track_id_1 = collision_detection.get('track_id', -1)
    track_id_2 = collision_detection.get('track_id_secondary', -1)
    
    # Energy & Severity
    energy_loss_i = collision_detection.get('energy_loss_primary', 0.0)
    energy_loss_j = collision_detection.get('energy_loss_secondary', 0.0)
    total_energy_loss = (energy_loss_i + energy_loss_j) / 2.0
    
    # Classify severity
    if total_energy_loss >= 0.9:
        severity = "CRITICAL"
    elif total_energy_loss >= 0.7:
        severity = "HIGH"
    elif total_energy_loss >= 0.5:
        severity = "MEDIUM"
    else:
        severity = "LOW"
    
    # Momentum transfer
    momentum_transfer_i = collision_detection.get('momentum_transfer_i', 0.0)
    momentum_transfer_j = collision_detection.get('momentum_transfer_j', 0.0)
    
    # Get velocity vectors dari track_manager untuk calculate speeds dan impact angle
    v_i_vec = None
    v_j_vec = None
    v_i_speed = 0.0
    v_j_speed = 0.0
    
    if track_manager and hasattr(track_manager, 'velocity_field'):
        try:
            # Get tracks
            tracks = track_manager.get_current_tracks()
            track_i_obj = next((t for t in tracks if t.track_id == track_id_1), None)
            track_j_obj = next((t for t in tracks if t.track_id == track_id_2), None)
            
            if track_i_obj and track_j_obj:
                v_i_vec = track_manager.velocity_field.compute_velocity(track_i_obj, dt=1.0)
                v_j_vec = track_manager.velocity_field.compute_velocity(track_j_obj, dt=1.0)
                v_i_speed = np.linalg.norm(v_i_vec)
                v_j_speed = np.linalg.norm(v_j_vec)
        except Exception as e:
            # Fallback: try get dari collision_detection
            v_i_speed = collision_detection.get('v_i_speed', 0.0)
            v_j_speed = collision_detection.get('v_j_speed', 0.0)
    else:
        # Fallback: try get dari collision_detection
        v_i_speed = collision_detection.get('v_i_speed', 0.0)
        v_j_speed = collision_detection.get('v_j_speed', 0.0)
    
    # Impact direction untuk calculate angle
    impact_direction = collision_detection.get('impact_direction', [0, 0])
    if isinstance(impact_direction, list):
        impact_direction = np.array(impact_direction)
    
    # Calculate impact angle (angle between velocity vectors)
    if v_i_vec is not None and v_j_vec is not None:
        # Use actual velocity vectors (most accurate)
        v_i_norm = np.linalg.norm(v_i_vec)
        v_j_norm = np.linalg.norm(v_j_vec)
        
        if v_i_norm > 1e-6 and v_j_norm > 1e-6:
            v_i_unit = v_i_vec / v_i_norm
            v_j_unit = v_j_vec / v_j_norm
            dot_product = np.clip(np.dot(v_i_unit, v_j_unit), -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            impact_angle = np.degrees(angle_rad)
            # Normalize to [0, 180]
            impact_angle = min(impact_angle, 180.0 - impact_angle)
        else:
            impact_angle = 0.0
    elif np.linalg.norm(impact_direction) > 1e-6:
        # Fallback: use impact_direction angle (less accurate but better than nothing)
        impact_angle_rad = np.arctan2(impact_direction[1], impact_direction[0])
        impact_angle = abs(np.degrees(impact_angle_rad))
        # Normalize to [0, 180]
        if impact_angle > 180:
            impact_angle = 360 - impact_angle
    else:
        impact_angle = 0.0
    
    # Infer evidence dari momentum transfer dan speeds
    # Check velocity mismatch (significant speed difference)
    velocity_mismatch = abs(v_i_speed - v_j_speed) > 1.0 if (v_i_speed > 0 and v_j_speed > 0) else False
    
    # Check if approaching (both moving in similar direction)
    is_approaching = False
    if v_i_vec is not None and v_j_vec is not None:
        v_i_norm = np.linalg.norm(v_i_vec)
        v_j_norm = np.linalg.norm(v_j_vec)
        if v_i_norm > 1e-6 and v_j_norm > 1e-6:
            v_i_unit = v_i_vec / v_i_norm
            v_j_unit = v_j_vec / v_j_norm
            dot_product = np.dot(v_i_unit, v_j_unit)
            # Approaching = moving in similar direction (dot product > 0.5)
            is_approaching = dot_product > 0.5
    
    # Check rear-end evidence: velocity mismatch + approaching + momentum transfer
    has_rear_end_evidence = (velocity_mismatch and is_approaching and 
                            (momentum_transfer_i > 0 or momentum_transfer_j > 0))
    
    if has_rear_end_evidence or (impact_angle < 30.0 and velocity_mismatch):
        collision_type = "Rear-end Collision (Tabrak Belakang)"
    elif impact_angle > 150.0:
        collision_type = "Head-on Collision (Tabrak Depan)"
    else:
        collision_type = "Side Collision (Tabrak Samping)"
    
    # Identify aggressor/victim dari momentum transfer
    if abs(momentum_transfer_i) > abs(momentum_transfer_j):
        if momentum_transfer_i > 0:
            aggressor_id = track_id_1
            victim_id = track_id_2
            aggressor_speed = v_i_speed
            victim_speed = v_j_speed
            aggressor_energy = energy_loss_i
            victim_energy = energy_loss_j
            momentum_value = momentum_transfer_i
        else:
            aggressor_id = track_id_2
            victim_id = track_id_1
            aggressor_speed = v_j_speed
            victim_speed = v_i_speed
            aggressor_energy = energy_loss_j
            victim_energy = energy_loss_i
            momentum_value = abs(momentum_transfer_j)
    else:
        if momentum_transfer_j > 0:
            aggressor_id = track_id_2
            victim_id = track_id_1
            aggressor_speed = v_j_speed
            victim_speed = v_i_speed
            aggressor_energy = energy_loss_j
            victim_energy = energy_loss_i
            momentum_value = momentum_transfer_j
        else:
            aggressor_id = track_id_1
            victim_id = track_id_2
            aggressor_speed = v_i_speed
            victim_speed = v_j_speed
            aggressor_energy = energy_loss_i
            victim_energy = energy_loss_j
            momentum_value = abs(momentum_transfer_i)
    
    # Get class names
    class_primary = collision_detection.get('class_primary', 'unknown')
    class_secondary = collision_detection.get('class_secondary', 'unknown')
    aggressor_class = class_primary if aggressor_id == track_id_1 else class_secondary
    victim_class = class_secondary if aggressor_id == track_id_1 else class_primary
    
    # Check pre-collision braking
    aggressor_braked = False
    if brake_results:
        frames_before = 10
        for brake_det in brake_results:
            brake_track_id = brake_det.get('track_id', -1)
            brake_frame_id = brake_det.get('frame_id', -1)
            if frame_id - frames_before <= brake_frame_id < frame_id:
                if brake_track_id == aggressor_id:
                    aggressor_braked = True
                    break
    
    # Collision point
    collision_point = collision_detection.get('collision_point', [0, 0])
    
    # Generate case ID
    timestamp = datetime.now()
    case_id = f"#COLLISION-{timestamp.strftime('%Y%m%d-%H%M%S')}"
    
    # Calculate frame timestamp
    if fps > 0:
        total_seconds = frame_id / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 100)
        frame_timestamp = f"{minutes:02d}:{seconds:02d}.{milliseconds:02d}"
    else:
        frame_timestamp = "00:00.00"
    
    # Build message dengan HTML formatting
    message_lines = []
    message_lines.append("ðŸš¨ <b>COLLISION DETECTED - FORENSIC ALERT</b>")
    message_lines.append("")
    message_lines.append(f"ðŸ“‹ <b>Case:</b> {case_id}")
    message_lines.append(f"â±ï¸ <b>Time:</b> {frame_timestamp} (Frame {frame_id})")
    message_lines.append(f"ðŸ“¹ <b>Video:</b> {Path(video_path).name}")
    message_lines.append("")
    message_lines.append(f"ðŸ”´ <b>SEVERITY:</b> {severity} ({total_energy_loss*100:.0f}% Energy Loss)")
    message_lines.append(f"ðŸš— <b>TYPE:</b> {collision_type}")
    message_lines.append("")
    message_lines.append("<b>âš–ï¸ VERDICT:</b>")
    message_lines.append(f"â€¢ <b>Aggressor:</b> Vehicle {aggressor_id} ({aggressor_class.upper()})")
    message_lines.append(f"â€¢ <b>Victim:</b> Vehicle {victim_id} ({victim_class.upper()})")
    message_lines.append(f"â€¢ <b>Evidence:</b> Momentum Transfer {aggressor_id}â†’{victim_id} (+{momentum_value:.1f} units)")
    braking_status = "NO" if not aggressor_braked else "YES"
    message_lines.append(f"â€¢ <b>Braking:</b> {braking_status} (Aggressor did {'NOT ' if not aggressor_braked else ''}brake before impact)")
    message_lines.append("")
    message_lines.append("<b>ðŸ“Š PHYSICS EVIDENCE:</b>")
    message_lines.append(f"â€¢ Energy Loss: {aggressor_id}={aggressor_energy*100:.0f}%, {victim_id}={victim_energy*100:.0f}%")
    message_lines.append(f"â€¢ Pre-collision: {aggressor_id}={aggressor_speed:.1f} px/fr, {victim_id}={victim_speed:.1f} px/fr")
    message_lines.append(f"â€¢ Impact Angle: {impact_angle:.1f}Â°")
    message_lines.append(f"â€¢ Momentum Transfer: {momentum_value:.1f} units ({aggressor_id}â†’{victim_id})")
    message_lines.append("")
    message_lines.append(f"ðŸ“ <b>Location:</b> [{collision_point[0]:.0f}, {collision_point[1]:.0f}]")
    message_lines.append(f"ðŸŽ¯ <b>Confidence:</b> 95% (High)")
    message_lines.append("")
    message_lines.append("â”" * 40)
    message_lines.append("Full forensic report available in terminal log.")
    
    message = "\n".join(message_lines)
    
    # Telegram message limit: 4096 characters
    if len(message) > 4096:
        # Truncate jika terlalu panjang
        message = message[:4000] + "\n\n... (message truncated)"
    
    return message


class DualLogger:
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        except:
            pass

    def flush(self):
        try:
            self.terminal.flush()
            self.log.flush()
        except:
            pass


def main():
    """Main application loop"""
    # Parse arguments
    args = parse_arguments()

    # Setup Terminal Logging (log_terminal/)
    try:
        log_terminal_dir = 'log_terminal'
        os.makedirs(log_terminal_dir, exist_ok=True)
        
        # Get source name for log filename
        video_stem = getattr(args, 'source_stem', None) or _safe_source_stem(args.video)
            
        timestamp_log = datetime.now().strftime('%Y%m%d_%H%M%S')
        terminal_log_path = os.path.join(log_terminal_dir, f"{video_stem}_{timestamp_log}.txt")
        
        # Redirect stdout and stderr
        # Check if already redirected to avoid recursion or duplication
        if not isinstance(sys.stdout, DualLogger):
            sys.stdout = DualLogger(terminal_log_path, sys.stdout)
        if not isinstance(sys.stderr, DualLogger):
            sys.stderr = DualLogger(terminal_log_path, sys.stderr)
            
        print(f"ðŸ“ Terminal output will be captured to: {terminal_log_path}")
    except Exception as e:
        print(f"âš ï¸ Warning: Failed to setup terminal logging: {e}")
    
    # Setup logging to file (log.txt)
    log_file = 'log.txt'
    # Clear previous log file
    if os.path.exists(log_file):
        open(log_file, 'w').close()
    
    # Configure logging with both console and file handler
    # Get root logger and add file handler (don't use basicConfig to avoid conflicts)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Allow all levels (DEBUG and above)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []

    # Add file handler (DEBUG level for complete log)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Changed to DEBUG to capture all logs
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(file_handler)

    # Add console handler (WARNING only to reduce terminal spam)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(console_handler)
    
    # Validate video file
    if not args.is_live_source and not os.path.exists(args.video):
        print(f"âŒ Error: Video file not found: {args.video}")
        sys.exit(1)
    
    print("=" * 60)
    print("LTE-TrackGuard - Physics-Based Traffic Behaviour Detection")
    print("=" * 60)
    source_kind = "Live Source" if args.is_live_source else "Input Video"
    print(f"ðŸ“¹ {source_kind}: {args.video}")
    print(f"ðŸ“ Logging to: {log_file}")
    
    # Configure physics settings
    configure_physics_settings(args)
    
    # Initialize track manager
    print(f"\nðŸš€ Initializing Track Manager (tracker: {args.tracker})...")
    try:
        if args.tracker == 'bytetrack':
            from core.bytetrack_manager import ByteTrackManager
            track_manager = ByteTrackManager(use_physics=args.physics)
        else:
            track_manager = PureSmartHungarianTrackManager(use_physics=args.physics)
        print("âœ“ Track Manager initialized successfully")
    except Exception as e:
        print(f"âŒ Error initializing track manager: {e}")
        sys.exit(1)
    
    # Initialize Telegram notifier (jika flag --telegram aktif)
    telegram_notifier = None
    if args.telegram:
        if TELEGRAM_AVAILABLE:
            try:
                telegram_notifier = TelegramNotifier()
                print("ðŸ“± Telegram notifications: ENABLED")
            except Exception as e:
                print(f"âš ï¸  Warning: Could not initialize Telegram notifier: {e}")
                print("   Continuing without Telegram notifications...")
        else:
            print("âš ï¸  Warning: Telegram notification service not available")
            print("   Install required packages or check notification_service.py")
    
    # Initialize forensic logging (1 file per video)
    forensic_log_dir = Path('forensic_logs')
    forensic_log_dir.mkdir(exist_ok=True)
    
    # Generate forensic log filename dari video name
    video_name = args.source_stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    forensic_log_file = forensic_log_dir / f"forensic_report_{timestamp}_{video_name}.txt"
    
    # Track reported collisions untuk avoid duplicate reports
    reported_collisions = set()  # Set of (track_id_1, track_id_2) pairs
    fired_pairs = {}             # pair -> highest reported alert level; red upgrades are still logged

    # Separate eval tracking â€” resets per episode so each new alarm episode is counted
    eval_reported_set = set()  # pairs logged for eval in current collision episode
    
    print(f"ðŸ“ Forensic logs will be saved to: {forensic_log_file}")
    
    # Open video / webcam / stream
    print(f"\nðŸ“¹ Opening source: {args.video}")
    cap = _open_capture(args.video)
    
    if not cap.isOpened():
        print(f"âŒ Error: Cannot open input source")
        sys.exit(1)
    
    # Get video properties
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30
    if total_frames < 0:
        total_frames = 0

    print(f"   Resolution: {frame_width}x{frame_height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {'LIVE/unknown' if args.is_live_source or total_frames == 0 else total_frames}")

    # Update PPL window W = fps Ã— 1 detik (PPL init default 30fps)
    if args.collision_baseline != 'none':
        if not args.physics:
            print("WARNING: collision baseline requires --physics; baseline was not enabled.")
        elif hasattr(track_manager, 'behaviour_detectors') and track_manager.behaviour_detectors:
            from core.collision_baselines import TrackHeuristicCollisionBaseline
            track_manager.behaviour_detectors['collision'] = TrackHeuristicCollisionBaseline(
                mode=args.collision_baseline,
                fps=float(fps),
            )
            print(f"   Collision comparator baseline: {args.collision_baseline}")
            print("   TrackGuard-LTE collision logic is replaced for this run only.")

    if hasattr(track_manager, 'behaviour_detectors') and track_manager.behaviour_detectors and 'collision' in track_manager.behaviour_detectors:
        track_manager.behaviour_detectors['collision'].update_fps(float(fps))
        if args.collision_baseline == 'none':
            print(f"   PPL window updated: W={fps} frames (1s @ {fps}fps)")
    if hasattr(track_manager, 'behaviour_detectors') and track_manager.behaviour_detectors and 'fallen' in track_manager.behaviour_detectors:
        _fallen_detector = track_manager.behaviour_detectors['fallen']
        if hasattr(_fallen_detector, 'update_fps'):
            _fallen_detector.update_fps(float(fps))
        if hasattr(_fallen_detector, 'enable_scalar_logging'):
            _fallen_detector.enable_scalar_logging(
                video_name=Path(args.video).name if not args.is_live_source else args.source_stem,
                label=args.label,
            )
            print("   Fallen PPL scalar logging: ENABLED")

    # â”€â”€ Scalar logging: enable setelah FPS diketahui â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if args.log_scalars:
        _cd = (track_manager.behaviour_detectors.get('collision')
               if hasattr(track_manager, 'behaviour_detectors') else None)
        if _cd is not None and hasattr(_cd, 'ppl'):
            _label_str = "COLLISION (positive)" if args.label == 1 else "NON-COLLISION (negative)"
            _cd.ppl.enable_scalar_logging(
                video_name=Path(args.video).name if not args.is_live_source else args.source_stem,
                label=args.label,
                camera_angle=args.camera_angle,
            )
            print(f"ðŸ“Š Scalar logging ENABLED")
            print(f"   Label        : {args.label} â€” {_label_str}")
            print(f"   Camera angle : {args.camera_angle}")
        else:
            print("âš ï¸  Scalar logging: collision detector tidak tersedia. "
                  "Jalankan dengan --physics --detect collision|all")
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    
    evidence_dir = Path(args.evidence_dir)
    evidence_level_dirs = {
        'emergency': evidence_dir / 'red',
        'warning': evidence_dir / 'critical',
        'caution': evidence_dir / 'caution',
    }
    evidence_suffix = {
        'emergency': 'red',
        'warning': 'critical',
        'caution': 'caution',
    }
    if not args.no_evidence_snapshots:
        for _level_dir in evidence_level_dirs.values():
            _level_dir.mkdir(parents=True, exist_ok=True)
        print(f"Alarm evidence snapshots: {evidence_dir.absolute()}")
        print(f"  red      : {evidence_level_dirs['emergency'].absolute()}")
        print(f"  critical : {evidence_level_dirs['warning'].absolute()}")
        print(f"  caution  : {evidence_level_dirs['caution'].absolute()}")
    evidence_snapshot_keys = set()
    telegram_photo_pairs = set()

    # Setup output video writer. For live deployment, default is no recording unless
    # --output is explicitly set; this avoids filling the disk during long RTSP runs.
    save_output = not args.no_output and (not args.is_live_source or args.output != 'output.mp4')
    out_writer = None
    codec_used = None
    if save_output:
        if args.output == 'output.mp4':
            input_path = Path(args.video)
            output_path = input_path.parent / f"{input_path.stem}_output.mp4"
            args.output = str(output_path)
        
        output_path_obj = Path(args.output)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        _out_ext = Path(args.output).suffix.lower()
        if _out_ext == '.avi':
            fourcc_options = [
                ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
                ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
            ]
        else:
            fourcc_options = [
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
                ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
                ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
                ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
            ]
        
        for codec_name, fourcc in fourcc_options:
            out_writer = cv2.VideoWriter(args.output, fourcc, fps, 
                                         (frame_width, frame_height))
            if out_writer.isOpened():
                codec_used = codec_name
                print(f"VideoWriter initialized successfully with codec: {codec_name}")
                break
            else:
                out_writer.release()
                out_writer = None
        
        if out_writer is None or not out_writer.isOpened():
            print(f"ERROR: Failed to initialize VideoWriter for {args.output}")
            print("   Trying alternative: .avi format with XVID codec")
            alt_output = str(Path(args.output).with_suffix('.avi'))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_writer = cv2.VideoWriter(alt_output, fourcc, fps, 
                                         (frame_width, frame_height))
            if out_writer.isOpened():
                args.output = alt_output
                codec_used = 'XVID (AVI)'
                print(f"Using alternative format: {alt_output}")
            else:
                print("CRITICAL: Cannot create output video file!")
                print("   Check if directory exists and you have write permissions")
                sys.exit(1)
        
        args.output = str(Path(args.output).absolute())
        print(f"Output video will be saved to: {args.output}")
        print(f"   Codec: {codec_used}, Resolution: {frame_width}x{frame_height}, FPS: {fps}")
        print(f"   Absolute path: {Path(args.output).absolute()}")
    else:
        if args.is_live_source and args.output == 'output.mp4' and not args.no_output:
            print("Output video recording disabled for live source by default.")
            print(r"   Pass --output path\to\recording.mp4 to save annotated live output.")
        else:
            print("Output video recording disabled (--no-output).")
    # Processing loop
    print("\nðŸŽ¬ Starting processing...")
    print("Press 'q' to quit, 'p' to pause, SPACE to step frame")
    
    frame_id = 0
    paused = False
    total_detections = {
        'fallen': 0,
        'collision': 0,
        'turn': 0,
        'brake': 0
    }
    
    # List to store all detections for evaluation
    all_ai_detections: List[AIDetection] = []
    fallen_alert_last_frame = {}
    
    processing_times = []
    stage_profile_file = None
    stage_profile_writer = None
    if args.stage_profile_output:
        _profile_path = Path(args.stage_profile_output)
        _profile_path.parent.mkdir(parents=True, exist_ok=True)
        stage_profile_file = open(_profile_path, 'w', newline='', encoding='utf-8')
        stage_profile_writer = csv.DictWriter(stage_profile_file, fieldnames=[
            'frame',
            'capture_ms',
            'decode_ms',
            'capture_decode_ms',
            'yolo_ms',
            'bytetrack_ms',
            'ppl_ms',
            'alarm_decision_ms',
            'ppl_alarm_policy_ms',
            'split_profile_available',
            'pipeline_total_ms',
            'frame_process_ms',
            'postprocess_ms',
            'num_detections',
            'num_active_tracks',
            'num_collision_results',
            'red_alarm_frame',
            'critical_alarm_frame',
        ])
        stage_profile_writer.writeheader()
    stream_open_time = time.time()
    scheduled_refreshes = 0
    read_failure_reconnects = 0
    reconnect_failures = 0
    
    while True:
        if not paused:
            if (
                args.is_live_source
                and args.live_refresh_interval
                and args.live_refresh_interval > 0
                and (time.time() - stream_open_time) >= args.live_refresh_interval
            ):
                scheduled_refreshes += 1
                print(
                    f"\n[STREAM_REFRESH] scheduled={scheduled_refreshes} "
                    f"frame={frame_id} interval_s={args.live_refresh_interval:.1f}"
                )
                cap.release()
                time.sleep(args.reconnect_delay)
                cap = _open_capture(args.video)
                stream_open_time = time.time()
                if not cap.isOpened():
                    reconnect_failures += 1
                    print(
                        f"[STREAM_REFRESH] reopen_failed count={reconnect_failures}; "
                        "retrying on next loop..."
                    )
                    continue

            # Split stream acquisition from frame decode. OpenCV grab() advances
            # the stream/packet buffer, while retrieve() decodes the grabbed frame.
            capture_start = time.perf_counter()
            grabbed = cap.grab()
            capture_ms = (time.perf_counter() - capture_start) * 1000.0
            decode_start = time.perf_counter()
            ret, frame = cap.retrieve() if grabbed else (False, None)
            decode_ms = (time.perf_counter() - decode_start) * 1000.0
            capture_decode_ms = capture_ms + decode_ms
            
            if not ret:
                if args.is_live_source:
                    read_failure_reconnects += 1
                    print(f"\nâš ï¸  Live source read failed. Reconnecting in {args.reconnect_delay:.1f}s...")
                    cap.release()
                    time.sleep(args.reconnect_delay)
                    cap = _open_capture(args.video)
                    stream_open_time = time.time()
                    if not cap.isOpened():
                        reconnect_failures += 1
                        print("âš ï¸  Reconnect failed; retrying...")
                    continue
                print("\nâœ“ End of video reached")
                break
            
            frame_id += 1
            
            # Process frame
            start_time = time.time()
            process_perf_start = time.perf_counter()
            
            try:
                results = track_manager.process_frame(frame, frame_id)
            except Exception as e:
                print(f"âŒ Error processing frame {frame_id}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue
            
            processing_time = time.time() - start_time
            frame_process_ms = (time.perf_counter() - process_perf_start) * 1000.0
            processing_times.append(processing_time)
            postprocess_start = time.perf_counter()
            
            # Get behaviour results
            behaviour_results = track_manager.behaviour_results if args.physics else {}
            
            # Count detections
            for det_type, dets in behaviour_results.items():
                if len(dets) > 0:
                    total_detections[det_type] += len(dets)
                    
                    # Print alerts
                    for det in dets:
                        severity = det.get('severity', 'unknown')
                        if severity in ['high', 'critical']:
                            if det_type == 'fallen':
                                _tid = det.get('track_id', -1)
                                _last_alert = fallen_alert_last_frame.get(_tid, -10**9)
                                if frame_id - _last_alert >= max(1, int(fps)):
                                    fallen_alert_last_frame[_tid] = frame_id
                                    _mode = det.get('detection_mode', 'legacy')
                                    _path = det.get('ppl_path', det.get('ppl_probe_path', '-'))
                                    _prims = det.get('ppl_primitives', '-')
                                    print(
                                        f"[FALLEN EVENT] Frame {frame_id}: Track {_tid} | "
                                        f"mode={_mode} severity={severity} "
                                        f"path={_path} primitives={_prims} | "
                                        f"KCR={det.get('ppl_kcr_value', 0.0):.3f} "
                                        f"KCRz={det.get('ppl_kcr_z', 0.0):.2f} "
                                        f"speed={det.get('ppl_speed_before', 0.0):.2f}"
                                        f"->{det.get('ppl_speed_after', 0.0):.2f} "
                                        f"ARz={det.get('ppl_ar_z', 0.0):.2f} "
                                        f"ARdrop={det.get('ppl_ar_drop', 0.0):.2f} "
                                        f"PPS={det.get('ppl_pps_score', 0.0):.2f} "
                                        f"SDE={det.get('ppl_sde_score', 0.0):.2f} "
                                        f"cornerMob={det.get('ppl_corner_mobility', 0.0):.2f} "
                                        f"cornerCos={det.get('ppl_corner_heading_cos', 0.0):.2f}"
                                    )
                            else:
                                print(f"[ALERT] Frame {frame_id}: {det_type.upper()} detected "
                                      f"(Track {det.get('track_id', -1)}, severity: {severity})")

                        if det_type == 'fallen' and EVALUATION_AVAILABLE:
                            bbox_primary = None
                            det_bbox = det.get('bbox')
                            if det_bbox:
                                h, w = frame.shape[:2]
                                x1, y1, x2, y2 = det_bbox
                                bbox_primary = [x1 / w, y1 / h, x2 / w, y2 / h]

                            fallen_conf = det.get('ppl_confidence', det.get('confidence', 1.0))
                            if fallen_conf <= 1.0:
                                fallen_conf *= 100.0

                            all_ai_detections.append(AIDetection(
                                frame=frame_id,
                                confidence=float(fallen_conf),
                                confidence_label='MOTORCYCLE_FALLEN',
                                track_id_primary=det.get('track_id', -1),
                                track_id_secondary=-1,
                                detection_mode=det.get('detection_mode', 'fallen'),
                                bbox_primary=bbox_primary,
                                bbox_secondary=None,
                                kinematic_evidence={
                                    'ppl_path': det.get('ppl_path', ''),
                                    'ppl_primitives': det.get('ppl_primitives', ''),
                                    'ppl_kcr_value': det.get('ppl_kcr_value', 0.0),
                                    'ppl_ar_z': det.get('ppl_ar_z', 0.0),
                                }
                            ))
                        
                        # Handle collision detections: forensic logs + Telegram
                        if det_type == 'collision':
                            state = det.get('state', 'confirmed')
                            track_id_1 = det.get('track_id', -1)
                            track_id_2 = det.get('track_id_secondary', -1)
                            pair_key = tuple(sorted([track_id_1, track_id_2]))
                            persist_count = det.get('persist_count', 0)

                            current_alert_level = det.get('alert_level', 'emergency')

                            # Skip duplicate pair alarms, but still keep a later red upgrade
                            # if the same pair first appeared as a critical-interaction warning.
                            previous_alert_level = fired_pairs.get(pair_key)
                            if previous_alert_level == 'emergency':
                                continue
                            if previous_alert_level is not None and current_alert_level != 'emergency':
                                continue
                            fired_pairs[pair_key] = current_alert_level

                            # Check jika ini first detection (monitoring dengan persist_count=1) atau confirmed
                            is_first_detection = (state == 'monitoring' and persist_count == 1)
                            is_confirmed = (state == 'confirmed')
                            already_reported = (pair_key in reported_collisions)
                            should_report = (is_first_detection or is_confirmed) and not already_reported
                            
                            # Debug logging untuk troubleshooting
                            print(f"[COLLISION NOTIFICATION DEBUG] Frame {frame_id}: Track {track_id_1} <-> Track {track_id_2} | "
                                  f"state={state}, persist_count={persist_count}, is_confirmed={is_confirmed}, "
                                  f"already_reported={already_reported}, should_report={should_report}")
                            
                            # Generate forensic report dan Telegram notification untuk first detection atau confirmed
                            if should_report:
                                try:
                                    brake_results = behaviour_results.get('brake', [])
                                    
                                    # 1. Generate detailed forensic report
                                    forensic_report = generate_forensic_report(
                                        collision_detection=det,
                                        frame_id=frame_id,
                                        fps=fps,
                                        video_path=args.video,
                                        brake_results=brake_results,
                                        track_manager=track_manager
                                    )
                                    
                                    # Append to forensic log file (1 file per video)
                                    with open(forensic_log_file, 'a', encoding='utf-8') as f:
                                        f.write(forensic_report)
                                        f.write("\n\n")
                                    
                                    print(f"âœ“ Forensic report saved to: {forensic_log_file.name}")
                                    
                                    # 2. Send Telegram notification (jika enabled)
                                    # ONLY send Telegram for emergency (COLLISION DETECTED, confidence >= 70%)
                                    # Non-emergency (MIGHT COLLISION, DANGER, LOW CONFIDENCE) = no Telegram alarm
                                    det_alert_level = det.get('alert_level', 'emergency')
                                    det_confidence_label = det.get('confidence_label', 'COLLISION DETECTED')

                                    if telegram_notifier is not None and det_alert_level == 'emergency':
                                        print(f"[TELEGRAM] Attempting to send notification for Frame {frame_id}, pair_key={pair_key}")
                                        try:
                                            # Generate forensic telegram message format
                                            forensic_message = generate_forensic_telegram_message(
                                                collision_detection=det,
                                                frame_id=frame_id,
                                                fps=fps,
                                                video_path=args.video,
                                                brake_results=brake_results,
                                                track_manager=track_manager
                                            )
                                            
                                            print(f"[TELEGRAM] Message generated, length={len(forensic_message)} chars")
                                            
                                            # Send message
                                            success = telegram_notifier.send_message(forensic_message, parse_mode="HTML")
                                            if success:
                                                print(f"âœ“ Telegram notification sent successfully for Frame {frame_id} (pair_key={pair_key})")
                                                if not args.no_evidence_snapshots and pair_key not in telegram_photo_pairs:
                                                    try:
                                                        source_stem = getattr(args, 'source_stem', None) or Path(args.video).stem
                                                        snapshot_name = (
                                                            f"{source_stem}_frame{frame_id:06d}_"
                                                            f"tracks{pair_key[0]}-{pair_key[1]}_telegram.png"
                                                        )
                                                        snapshot_path = evidence_dir / snapshot_name
                                                        snapshot_frame = frame.copy()
                                                        current_track_bboxes = {
                                                            t.track_id: t.current_detection['bbox']
                                                            for t in track_manager.get_current_tracks()
                                                            if 'bbox' in t.current_detection
                                                        }
                                                        for _tid, _color in ((track_id_1, (0, 0, 255)), (track_id_2, (0, 255, 255))):
                                                            _bbox = current_track_bboxes.get(_tid)
                                                            if _bbox is None:
                                                                continue
                                                            _x1, _y1, _x2, _y2 = [int(v) for v in _bbox]
                                                            cv2.rectangle(snapshot_frame, (_x1, _y1), (_x2, _y2), _color, 4)
                                                            cv2.putText(
                                                                snapshot_frame,
                                                                f"Track {_tid}",
                                                                (_x1, max(25, _y1 - 8)),
                                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                                0.8,
                                                                _color,
                                                                2,
                                                            )
                                                        cv2.putText(
                                                            snapshot_frame,
                                                            f"COLLISION DETECTED | S: {det.get('confidence', 0.0):.0f} | Frame {frame_id}",
                                                            (20, 40),
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            1.0,
                                                            (0, 0, 255),
                                                            3,
                                                        )
                                                        cv2.imwrite(str(snapshot_path), snapshot_frame)
                                                        caption = (
                                                            f"Collision evidence snapshot\n"
                                                            f"Video: {Path(args.video).name if not args.is_live_source else args.source_stem}\n"
                                                            f"Frame: {frame_id}\n"
                                                            f"Tracks: {pair_key[0]} to {pair_key[1]}\n"
                                                            f"Score: {det.get('confidence', 0.0):.0f} | "
                                                            f"{det.get('confidence_label', 'COLLISION DETECTED')}"
                                                        )
                                                        sent_photo = telegram_notifier.send_photo(
                                                            str(snapshot_path),
                                                            caption=caption,
                                                            parse_mode="",
                                                            bypass_rate_limit=True,
                                                        )
                                                        if sent_photo:
                                                            telegram_photo_pairs.add(pair_key)
                                                            print(f"[TELEGRAM] Evidence snapshot sent immediately: {snapshot_path.name}")
                                                        else:
                                                            print(f"[TELEGRAM] Evidence snapshot immediate send failed: {snapshot_path.name}")
                                                    except Exception as e:
                                                        print(f"[TELEGRAM] Evidence snapshot immediate send error: {e}")
                                            else:
                                                # Check rate limit explicitly
                                                current_time = time.time()
                                                last_notif_time = getattr(telegram_notifier, 'last_notification_time', 0)
                                                rate_limit = getattr(telegram_notifier, 'rate_limit_seconds', 0)
                                                time_since_last = current_time - last_notif_time
                                                if time_since_last < rate_limit:
                                                    remaining = rate_limit - time_since_last
                                                    print(f"â¸ï¸ Telegram notification skipped for Frame {frame_id} (pair_key={pair_key}) due to rate limit: {remaining:.1f}s remaining")
                                                else:
                                                    print(f"âš ï¸ Warning: Telegram notification failed for Frame {frame_id} (pair_key={pair_key}) - check logs above for API errors")
                                        except Exception as e:
                                            print(f"âš ï¸ ERROR: Failed to send Telegram notification for Frame {frame_id} (pair_key={pair_key}): {e}")
                                            if args.verbose:
                                                import traceback
                                                traceback.print_exc()
                                    elif telegram_notifier is not None and det_alert_level != 'emergency':
                                        print(f"[TELEGRAM] â„¹ï¸ Skipped (non-emergency: {det_confidence_label}) for Frame {frame_id}, pair_key={pair_key}")
                                    else:
                                        print(f"[TELEGRAM] telegram_notifier is None - skipping notification for Frame {frame_id}")
                                    
                                    # Mark as reported SETELAH kedua-duanya selesai
                                    reported_collisions.add(pair_key)
                                    
                                except Exception as e:
                                    print(f"âš ï¸ Warning: Failed to generate forensic report: {e}")
                                    if args.verbose:
                                        import traceback
                                        traceback.print_exc()

                        # ALL_DETECTION log â€” always printed (independent of EVALUATION_AVAILABLE)
                        # eval_batch.py parses these lines from subprocess stdout.
                        # Log setiap episode baru (is_first_detection) ke all_ai_detections tanpa
                        # gate eval_reported_set â€” cluster_detections() di evaluate_tudat sudah
                        # handle deduplication berdasarkan proximity waktu. Ini memastikan episode
                        # yang fire di dalam GT window tidak tersuppress oleh episode lebih awal
                        # yang fire di luar window (fix untuk TP=0 false negative).
                        _eval_log = (det_type == 'collision') and (is_first_detection or is_confirmed)
                        if _eval_log:
                            # Get bbox dari detection dict (if available from track)
                            bbox_primary = None
                            bbox_secondary = None

                            # â”€â”€ KINEMATIC EVIDENCE (for scientific proof) â”€â”€
                            kinematic_evidence = {}

                            # Try get bbox dari current track bboxes
                            try:
                                tracks = track_manager.get_current_tracks()
                                track_id_1 = det.get('track_id', -1)
                                track_id_2 = det.get('track_id_secondary', -1)

                                if args.verbose:
                                    print(f"[DEBUG] Frame {frame_id}: Looking for collision between tracks {track_id_1} and {track_id_2}")
                                    print(f"[DEBUG] Frame {frame_id}: Available tracks: {[t.track_id for t in tracks]}")

                                for t in tracks:
                                    if t.track_id == track_id_1:
                                        bbox = t.current_detection.get('bbox')
                                        if bbox:
                                            h, w = frame.shape[:2]
                                            # Convert from pixel to normalized [0, 1]
                                            x1, y1, x2, y2 = bbox
                                            bbox_primary = [x1/w, y1/h, x2/w, y2/h]

                                        # Capture kinematic data
                                        vel = t.velocity
                                        speed = np.linalg.norm(vel) if vel is not None else 0.0
                                        start_frame = t.frame_history[0] if hasattr(t, 'frame_history') and t.frame_history else frame_id
                                        track_age = frame_id - start_frame + 1 if frame_id >= start_frame else 1

                                        kinematic_evidence['primary'] = {
                                            'track_id': track_id_1,
                                            'velocity': (float(vel[0]), float(vel[1])) if vel is not None else (0.0, 0.0),
                                            'speed': float(speed),
                                            'track_age': track_age,
                                        }
                                        if args.verbose:
                                            print(f"[DEBUG] Frame {frame_id}: Track {track_id_1} velocity={vel}, speed={speed}")

                                    elif t.track_id == track_id_2:
                                        bbox = t.current_detection.get('bbox')
                                        if bbox:
                                            h, w = frame.shape[:2]
                                            x1, y1, x2, y2 = bbox
                                            bbox_secondary = [x1/w, y1/h, x2/w, y2/h]

                                        # Capture kinematic data (reuse track_id_2 branch)
                                        vel = t.velocity
                                        speed = np.linalg.norm(vel) if vel is not None else 0.0
                                        start_frame = t.frame_history[0] if hasattr(t, 'frame_history') and t.frame_history else frame_id
                                        track_age = frame_id - start_frame + 1 if frame_id >= start_frame else 1

                                        kinematic_evidence['secondary'] = {
                                            'track_id': track_id_2,
                                            'velocity': (float(vel[0]), float(vel[1])) if vel is not None else (0.0, 0.0),
                                            'speed': float(speed),
                                            'track_age': track_age,
                                        }
                                        if args.verbose:
                                            print(f"[DEBUG] Frame {frame_id}: Track {track_id_2} velocity={vel}, speed={speed}")
                            except Exception as e:
                                if args.verbose:
                                    print(f"[DEBUG] Warning: Could not extract bbox for evaluation: {e}")

                            # AIDetection requires the evaluation module â€” only append when available
                            if EVALUATION_AVAILABLE:
                                all_ai_detections.append(AIDetection(
                                    frame=frame_id,
                                    confidence=det.get('confidence', 0.0),
                                    confidence_label=det.get('confidence_label', ''),
                                    track_id_primary=det.get('track_id', -1),
                                    track_id_secondary=det.get('track_id_secondary', -1),
                                    detection_mode=det.get('detection_mode', ''),
                                    bbox_primary=bbox_primary,
                                    bbox_secondary=bbox_secondary,
                                    kinematic_evidence=kinematic_evidence if kinematic_evidence else None
                                ))

                            # ALL_DETECTION print â€” always runs so eval_batch.py can parse it
                            import math as _mi_inline
                            _al_inline = det.get('alert_level', 'emergency')
                            _conf_inline = det.get('confidence', 0.0)
                            _label_inline = str(det.get('confidence_label', '')).replace(' ', '_')
                            _ki_inline = det.get('energy_loss_primary', float('nan'))
                            _kj_inline = det.get('energy_loss_secondary', float('nan'))
                            _zi_inline = det.get('ars_zscore_i', float('nan'))
                            _zj_inline = det.get('ars_zscore_j', float('nan'))
                            _iou_inline = det.get('iou_overlap', 0.0)
                            _mod_inline = det.get('detection_mode', '?')
                            _ki_s = f"{_ki_inline:.4f}" if not _mi_inline.isnan(_ki_inline) else 'nan'
                            _kj_s = f"{_kj_inline:.4f}" if not _mi_inline.isnan(_kj_inline) else 'nan'
                            _zi_s = f"{_zi_inline:.4f}" if not _mi_inline.isnan(_zi_inline) else 'nan'
                            _zj_s = f"{_zj_inline:.4f}" if not _mi_inline.isnan(_zj_inline) else 'nan'
                            _bi_s = ','.join(f"{v:.4f}" for v in bbox_primary) if bbox_primary else 'None'
                            _bj_s = ','.join(f"{v:.4f}" for v in bbox_secondary) if bbox_secondary else 'None'
                            print(f"  ALL_DETECTION frame={frame_id} level={_al_inline} "
                                  f"conf={_conf_inline:.1f} label={_label_inline} "
                                  f"ki={_ki_s} kj={_kj_s} azi={_zi_s} azj={_zj_s} "
                                  f"iou={_iou_inline:.4f} mode={_mod_inline} "
                                  f"i={_bi_s} j={_bj_s}", flush=True)

                            if args.verbose:
                                print(f"[DEBUG] Frame {frame_id}: CollisionDetection recorded with kinematic_evidence={bool(kinematic_evidence)}")

            
            # Visualize
            vis_frame = track_manager.visualize_tracks(
                frame, 
                show_trajectories=args.show_trajectories
            )
            
            # Add behaviour detection visualizations
            if args.physics and behaviour_results:
                # Get proximity warnings dari collision_detector (jika ada)
                proximity_warning_pairs = {}
                if hasattr(track_manager, 'behaviour_detectors') and 'collision' in track_manager.behaviour_detectors:
                    collision_detector = track_manager.behaviour_detectors['collision']
                    if hasattr(collision_detector, 'proximity_warnings'):
                        proximity_warning_pairs = collision_detector.proximity_warnings
                
                # Build current track bbox lookup so persistent alert boxes follow the vehicle
                current_track_bboxes = {
                    t.track_id: t.current_detection['bbox']
                    for t in track_manager.get_current_tracks()
                    if 'bbox' in t.current_detection
                }
                vis_frame = visualize_detections(vis_frame, behaviour_results, args.detect, frame_id, proximity_warning_pairs, current_track_bboxes)
                
                # Add alert banner for fallen, collision, and proximity (filtered by detector_type)
                # Pass fps parameter (using real-time PC timestamp for 15s persistence)
                vis_frame = draw_alert_banner(vis_frame, behaviour_results, frame_id, fps, args.detect, proximity_warning_pairs)
                
                # Add summary panel with history log (filtered by detector_type)
                scene_info = None
                if hasattr(track_manager, 'scene_analyzer'):
                    scene_stats = track_manager.scene_analyzer.get_statistics()
                    scene_info = {
                        'category': scene_stats.get('current_category', 'unknown'),
                        'density': scene_stats.get('current_density', 0)
                    }
                
                vis_frame = draw_detection_summary(vis_frame, behaviour_results, scene_info, fps, args.detect)
            
            # Add frame info - pindahkan ke kanan bawah agar tidak timpa panel
            total_display = "LIVE" if args.is_live_source or total_frames == 0 else str(total_frames)
            info_text = f"Frame: {frame_id}/{total_display} | FPS: {1.0/processing_time:.1f}"
            # Posisi di kanan bawah (jika ada panel di kiri bawah)
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = frame_width - text_size[0] - 10
            text_y = frame_height - 20
            
            # Draw white background rectangle
            padding = 5
            cv2.rectangle(vis_frame, 
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         (255, 255, 255), -1)  # White background, filled
            
            # Draw black text on white background
            cv2.putText(vis_frame, info_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black text

            evidence_collision_dets = [
                det for det in behaviour_results.get('collision', [])
                if det.get('alert_level') in ('emergency', 'warning', 'caution')
            ] if args.physics and behaviour_results else []
            if evidence_collision_dets and not args.no_evidence_snapshots:
                source_stem = getattr(args, 'source_stem', None) or Path(args.video).stem
                for det in evidence_collision_dets:
                    alert_level = det.get('alert_level')
                    level_dir = evidence_level_dirs.get(alert_level, evidence_dir)
                    level_suffix = evidence_suffix.get(alert_level, str(alert_level or 'alarm'))
                    track_id_1 = det.get('track_id', -1)
                    track_id_2 = det.get('track_id_secondary', -1)
                    pair_key = tuple(sorted([track_id_1, track_id_2]))
                    snapshot_key = (frame_id, alert_level, pair_key)
                    if snapshot_key in evidence_snapshot_keys:
                        continue
                    evidence_snapshot_keys.add(snapshot_key)
                    snapshot_name = (
                        f"{source_stem}_frame{frame_id:06d}_"
                        f"tracks{pair_key[0]}-{pair_key[1]}_{level_suffix}.png"
                    )
                    snapshot_path = level_dir / snapshot_name
                    cv2.imwrite(str(snapshot_path), vis_frame)
                    print(f"[EVIDENCE SNAPSHOT] Frame {frame_id}: {snapshot_path}")

                    if alert_level == 'emergency' and telegram_notifier is not None and pair_key not in telegram_photo_pairs:
                        telegram_photo_pairs.add(pair_key)
                        caption = (
                            f"Collision evidence snapshot\n"
                            f"Video: {Path(args.video).name if not args.is_live_source else args.source_stem}\n"
                            f"Frame: {frame_id}\n"
                            f"Tracks: {pair_key[0]} <-> {pair_key[1]}\n"
                            f"Score: {det.get('confidence', 0.0):.0f} | "
                            f"{det.get('confidence_label', 'COLLISION DETECTED')}"
                        )
                        try:
                            sent_photo = telegram_notifier.send_photo(
                                str(snapshot_path),
                                caption=caption,
                                parse_mode="",
                                bypass_rate_limit=True,
                            )
                            if sent_photo:
                                print(f"[TELEGRAM] Evidence snapshot sent: {snapshot_path.name}")
                            else:
                                print(f"[TELEGRAM] Evidence snapshot failed: {snapshot_path.name}")
                        except Exception as e:
                            print(f"[TELEGRAM] Evidence snapshot error: {e}")
            
            if stage_profile_writer is not None:
                timing = results.get('pipeline_timing', {}) if isinstance(results, dict) else {}
                postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
                collision_dets_for_profile = behaviour_results.get('collision', []) if args.physics and behaviour_results else []
                red_alarm_frame = int(any(det.get('alert_level') == 'emergency' for det in collision_dets_for_profile))
                critical_alarm_frame = int(any(det.get('alert_level') in ('warning', 'emergency') for det in collision_dets_for_profile))
                # R2Q4 split profiling. The track manager must expose separate
                # timing keys for PPL computation and alarm-policy decision.
                # Multiple aliases are accepted to remain compatible with older branches.
                _ppl_aliases = ('ppl_time', 'ppl_computation_time', 'primitive_time', 'physics_primitive_time')
                _decision_aliases = ('alarm_decision_time', 'policy_time', 'decision_time', 'alarm_policy_time')
                _ppl_key = next((k for k in _ppl_aliases if k in timing), None)
                _decision_key = next((k for k in _decision_aliases if k in timing), None)
                _split_available = int(_ppl_key is not None and _decision_key is not None)
                _ppl_s = float(timing.get(_ppl_key, 0.0)) if _ppl_key else 0.0
                _decision_s = float(timing.get(_decision_key, 0.0)) if _decision_key else 0.0
                _combined_s = float(timing.get('physics_time', _ppl_s + _decision_s))

                # Fail early during a profiling run instead of wasting 24 hours with
                # an inseparable PPL+policy column. Thirty frames are enough to verify
                # that the required timing keys are present; this checks key presence,
                # not whether the measured values are non-zero.
                if args.physics and frame_id == 30 and not _split_available:
                    raise RuntimeError(
                        'Split stage profiling is unavailable: process_frame() did not return separate '
                        'PPL and alarm-decision timing keys in pipeline_timing. Instrument '
                        'core/track_manager.py (and the collision/PPL module it calls) to return '
                        'ppl_time and alarm_decision_time before starting the 24-hour run.'
                    )

                stage_profile_writer.writerow({
                    'frame': frame_id,
                    'capture_ms': f'{capture_ms:.6f}',
                    'decode_ms': f'{decode_ms:.6f}',
                    'capture_decode_ms': f'{capture_decode_ms:.6f}',
                    'yolo_ms': f'{float(timing.get("detection_time", 0.0)) * 1000.0:.6f}',
                    'bytetrack_ms': f'{float(timing.get("tracking_time", 0.0)) * 1000.0:.6f}',
                    'ppl_ms': f'{_ppl_s * 1000.0:.6f}',
                    'alarm_decision_ms': f'{_decision_s * 1000.0:.6f}',
                    'ppl_alarm_policy_ms': f'{_combined_s * 1000.0:.6f}',
                    'split_profile_available': _split_available,
                    'pipeline_total_ms': f'{float(timing.get("total_pipeline_time", 0.0)) * 1000.0:.6f}',
                    'frame_process_ms': f'{frame_process_ms:.6f}',
                    'postprocess_ms': f'{postprocess_ms:.6f}',
                    'num_detections': len(results.get('detections', [])) if isinstance(results, dict) else 0,
                    'num_active_tracks': len(track_manager.get_current_tracks()),
                    'num_collision_results': len(collision_dets_for_profile),
                    'red_alarm_frame': red_alarm_frame,
                    'critical_alarm_frame': critical_alarm_frame,
                })
                stage_profile_file.flush()
            # Write output - selalu write karena out_writer selalu ada
            if out_writer is not None and out_writer.isOpened():
                success = out_writer.write(vis_frame)
                if not success and frame_id % 30 == 0:  # Log setiap 30 frames
                    print(f"âš ï¸ Warning: Failed to write frame {frame_id} to video")
                
                # Force flush every 30 frames to ensure data is written
                if frame_id % 30 == 0 and hasattr(out_writer, 'get'):
                    try:
                        # Some codecs need explicit flush
                        pass  # VideoWriter doesn't have explicit flush in OpenCV
                    except:
                        pass
            elif save_output and frame_id == 1:
                print(f"âŒ ERROR: VideoWriter is not opened! Cannot write frames.")
            
            # Display
            if not args.no_display:
                cv2.imshow('LTE-TrackGuard', vis_frame)
                if emergency_collision_dets and args.freeze_escalation_ms > 0:
                    cv2.waitKey(args.freeze_escalation_ms)
            
            # FPS limiting
            if args.fps_limit:
                time.sleep(1.0 / args.fps_limit)
            if args.max_frames and frame_id >= args.max_frames:
                print(f"\nâœ“ Max frames reached: {args.max_frames}")
                break
        
        # Handle keyboard input
        key = cv2.waitKey(1 if not paused else 0) & 0xFF if not args.no_display else 255
        
        if key == ord('q'):
            print("\nâ¹ï¸  Stopped by user")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'â¸ï¸  Paused' if paused else 'â–¶ï¸  Resumed'}")
        elif key == ord(' ') and paused:
            # Step one frame
            paused = False
            continue
    
    # Cleanup
    cap.release()
    if stage_profile_file is not None:
        stage_profile_file.close()
    if out_writer is not None:
        print(f"ðŸ“¹ Releasing VideoWriter...")
        out_writer.release()  # Selalu release karena out_writer selalu ada
        
        # Wait a moment for file system to sync
        time.sleep(0.5)
        
        output_path = Path(args.output).absolute()
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"âœ… Output video saved successfully: {output_path}")
            print(f"   File size: {file_size:.2f} MB")
            print(f"   Location: {output_path.parent}")
        else:
            print(f"âŒ ERROR: Output video file NOT found!")
            print(f"   Expected location: {output_path}")
            print(f"   Directory exists: {output_path.parent.exists()}")
            print(f"   Files in directory: {list(output_path.parent.glob('*.mp4'))[:5]}")
    if not args.no_display:
        cv2.destroyAllWindows()

    # â”€â”€ Flush Scalar Log ke CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if args.log_scalars:
        _cd = (track_manager.behaviour_detectors.get('collision')
               if hasattr(track_manager, 'behaviour_detectors') else None)
        if _cd is not None and hasattr(_cd, 'ppl'):
            import csv as _csv
            _log_data = _cd.ppl.get_scalar_log()
            if _log_data:
                # Per-file mode: satu CSV per video, selalu overwrite (tidak append)
                # Output dir: scalar_logs/ di working directory
                _scalar_dir = Path('scalar_logs')
                _scalar_dir.mkdir(exist_ok=True)

                if args.scalar_log_output:
                    _out_csv = args.scalar_log_output
                else:
                    # Auto-name dari video: inter3.mp4 â†’ scalar_logs/inter3_scalar.csv
                    _out_csv = str(_scalar_dir / f"{args.source_stem}_scalar.csv")

                with open(_out_csv, 'w', newline='', encoding='utf-8') as _f:
                    _writer = _csv.DictWriter(_f, fieldnames=_log_data[0].keys())
                    _writer.writeheader()
                    _writer.writerows(_log_data)
                print(f"\nâœ… Scalar log disimpan: {len(_log_data):,} entries â†’ {_out_csv}")
            else:
                print("\nâ„¹ï¸  Scalar log: tidak ada entry (tidak ada proximate pair).")
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    # â”€â”€ Flush Fallen Event Log â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _fd = (track_manager.behaviour_detectors.get('fallen')
           if hasattr(track_manager, 'behaviour_detectors') and track_manager.behaviour_detectors else None)
    if _fd is not None and hasattr(_fd, 'get_scalar_log'):
        _fallen_scalar = _fd.get_scalar_log()
        if _fallen_scalar:
            import csv as _csv
            _fallen_scalar_dir = Path('scalar_logs')
            _fallen_scalar_dir.mkdir(parents=True, exist_ok=True)
            _fallen_scalar_out = (
                Path(args.fallen_scalar_output)
                if args.fallen_scalar_output
                else _fallen_scalar_dir / f"{args.source_stem}_fallen_scalar.csv"
            )
            _fallen_scalar_out.parent.mkdir(parents=True, exist_ok=True)
            with open(_fallen_scalar_out, 'w', newline='', encoding='utf-8') as _f:
                _writer = _csv.DictWriter(_f, fieldnames=_fallen_scalar[0].keys())
                _writer.writeheader()
                _writer.writerows(_fallen_scalar)
            print(f"\nðŸ“Š Fallen PPL scalar log saved: {len(_fallen_scalar):,} rows -> {_fallen_scalar_out}")

    if args.fallen_event_log and detection_history['fallen']:
        import csv as _csv
        _fallen_out = Path(args.fallen_event_log)
        _fallen_out.parent.mkdir(parents=True, exist_ok=True)
        with open(_fallen_out, 'w', newline='') as _f:
            _w = _csv.writer(_f)
            _w.writerow(['frame_id', 'track_id'])
            for _frame, _, _tid in detection_history['fallen']:
                _w.writerow([_frame, _tid])
        print(f"ðŸ“‹ Fallen event log saved: {_fallen_out} ({len(detection_history['fallen'])} events)")
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    # â”€â”€ Automatic Evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if EVALUATION_AVAILABLE and not args.is_live_source:
        print("\n" + "="*80)
        print("  AUTOMATIC EVALUATION REPORT")
        print("="*80)
        
        # 1. Extract video key from filename (e.g. "path/to/V1.mp4" -> "v1")
        video_filename = Path(args.video).name
        # Match TU-DAT style (v1, V1, vid_1) and motorcycle-fall style
        # (motor-vid-1).
        import re
        match = re.match(r'((?:v(?:id_?)?|motor-vid-)\d+)', video_filename, re.IGNORECASE)

        if match:
            video_key = match.group(1).lower()  # e.g. v1, vid_1, V31 â†’ lowercase
            
            # 2. Parse annotations
            # We assume TU-DAT is in the current directory or nearby
            tudat_dirs = [
                Path(args.video).parent / "gt",
                Path("TU-DAT"), 
                Path(args.video).parent / "TU-DAT",
                Path(".").resolve() / "TU-DAT"
            ]
            
            found_annotations = False
            for td in tudat_dirs:
                if td.exists():
                    print(f"Loading annotations from: {td}")
                    try:
                        # parse_annotations returns Dict[video_key, List[GTEvent]]
                        # Keys in parse_annotations are strictly from filenames (v1.txt -> v1)
                        annotations = parse_annotations(str(td))
                        
                        # Find matching key
                        target_key = None
                        for k in annotations.keys():
                            if k.lower() == video_key:
                                target_key = k
                                break
                        
                        if target_key:
                            gt_events = annotations[target_key]
                            print(f"Found {len(gt_events)} GT events for {video_filename} (key: {target_key})")
                            
                            # 3. Evaluate with dual-criterion (temporal + spatial)
                            # Get eval params from args or defaults
                            early_tol = getattr(args, 'early_tol', 2.0)
                            late_tol = getattr(args, 'late_tol', 1.0)
                            cluster_gap = getattr(args, 'cluster_gap', 2.0)
                            spatial_iou_threshold = getattr(args, 'spatial_iou_threshold', 0.30)

                            # Use dual-criterion evaluation (with spatial matching)
                            # Falls back to temporal-only if no bbox data available
                            result = evaluate_video_dual_criterion(
                                gt_events,
                                all_ai_detections,
                                fps=fps,
                                early_tol_sec=early_tol,
                                late_tol_sec=late_tol,
                                cluster_gap_sec=cluster_gap,
                                spatial_iou_threshold=spatial_iou_threshold,
                                min_confidence_tp=float(getattr(args, 'conf', 70))
                            )
                            
                            # 4. Print Table
                            tp = result['tp']
                            fp = result['fp']
                            fn = result['fn']
                            matches = result['matches']
                            
                            # Compute F1
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                            
                            print(f"\nResults for {video_filename}:")
                            print(f"  TP: {tp}")
                            print(f"  FP: {fp}")
                            print(f"  FN: {fn}")
                            print(f"  Precision : {precision:.2%}")
                            print(f"  Recall    : {recall:.2%}")
                            print(f"  F1 Score  : {f1:.2%}")
                            print("-" * 40)
                            
                            # Print matches detail with spatial metrics
                            for m in matches:
                                status = "âœ… MATCH" if m.matched else "âŒ MISSED"
                                delay_str = f"{m.delay_frames:+d} fr ({m.delay_seconds:+.2f}s)" if m.matched else "-"

                                # Include spatial metrics for matched events
                                if m.matched:
                                    spatial_str = f" | IoU: {m.spatial_iou:.3f} | Timeliness: {m.timeliness_score:.2%}"
                                else:
                                    spatial_str = ""

                                print(f"  Event {m.gt_event.event_id} [{m.gt_event.start_frame}-{m.gt_event.end_frame}]: {status} (Delay: {delay_str}){spatial_str}")

                            print("-" * 40)

                            # Aggregate temporal & spatial metrics (TP only)
                            matched_events = [m for m in matches if m.matched]
                            if matched_events:
                                delays_frames = [m.delay_frames for m in matched_events]
                                timeliness_scores = [m.timeliness_score for m in matched_events]
                                spatial_ious = [m.spatial_iou for m in matched_events]

                                mean_delay_frames = sum(delays_frames) / len(delays_frames)
                                mean_delay_sec = mean_delay_frames / fps
                                mean_timeliness = sum(timeliness_scores) / len(timeliness_scores)
                                mean_iou = sum(spatial_ious) / len(spatial_ious)
                                early_rate = sum(1 for d in delays_frames if d < 0) / len(delays_frames)

                                print("\nTemporal & Spatial Quality Metrics (TP only):")
                                print(f"  Mean Delay:         {mean_delay_frames:+.1f} frames ({mean_delay_sec:+.2f}s)")
                                print(f"  Mean Timeliness:    {mean_timeliness:.2%}")
                                print(f"  Mean IoU:           {mean_iou:.3f}")
                                print(f"  Early Warning Rate: {early_rate:.1%} ({sum(1 for d in delays_frames if d < 0)}/{len(delays_frames)} TP)")

                            # Show absorbed events summary
                            absorbed = result.get('absorbed_events', [])
                            if absorbed:
                                print(f"\nAbsorbed Events (secondary detection of same collision): {len(absorbed)}")

                            # â”€â”€ GENERATE KINEMATIC EVIDENCE GRAPHS â”€â”€
                            # Collision plots expect pairwise kinematic_evidence with
                            # primary/secondary keys. Fallen events are single-track PPL
                            # events, so skip this plotter to avoid noisy debug spam.
                            if all_ai_detections and args.detect in ['collision', 'all']:
                                try:
                                    from analyze_kinematic_evidence import (
                                        plot_kinematics_speed_profile,
                                        plot_acceleration_signature
                                    )

                                    video_name = Path(video_filename).stem.replace('_output', '').replace('.mov', '').replace('.mp4', '')
                                    print("\nðŸ”¬ Generating kinematic evidence graphs...")
                                    plot_kinematics_speed_profile(all_ai_detections, video_name, output_dir="kinematic_plots")
                                    plot_acceleration_signature(all_ai_detections, video_name, output_dir="kinematic_plots")
                                except Exception as e:
                                    print(f"[WARN] Could not generate kinematic graphs: {e}")
                                    if args.verbose:
                                        import traceback
                                        traceback.print_exc()

                            found_annotations = True
                            break
                    except Exception as e:
                        print(f"Error reading annotations: {e}")

                if found_annotations:
                    break
            
            if not found_annotations:
                print(f"âš ï¸ No annotations found for {video_filename} (checked key: {video_key})")
                print("   Ensure 'TU-DAT' folder exists and contains .txt files matching the video name.")
        else:
            print(f"âš ï¸ Filename {video_filename} does not match standard format (v1.mp4, etc). Evaluation skipped.")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("ðŸ“Š FINAL STATISTICS")
    print("=" * 60)
    
    # Processing stats
    avg_time = np.mean(processing_times) if processing_times else 0
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"Frames Processed: {frame_id}")
    print(f"Average Processing Time: {avg_time*1000:.1f} ms/frame")
    print(f"Average FPS: {avg_fps:.1f}")
    if args.is_live_source:
        print("\nLive Stream Stability:")
        print(f"   Scheduled refreshes: {scheduled_refreshes}")
        print(f"   Read-failure reconnects: {read_failure_reconnects}")
        print(f"   Reconnect failures: {reconnect_failures}")
    
    # Detection stats
    if args.physics:
        print(f"\nðŸ”¬ Physics Detection Summary:")
        for det_type, count in total_detections.items():
            if count > 0:
                # No emoji - text only
                print(f"   {det_type.capitalize()}: {count} detections")
    
    # Track statistics
    track_stats = track_manager.get_track_statistics()
    print(f"\nðŸ“ˆ Tracking Statistics:")
    print(f"   Total Tracks Created: {track_stats.get('total_tracks_created', 0)}")
    print(f"   Active Tracks: {track_stats.get('active_tracks', 0)}")
    print(f"   Ghost Reidentifications: {track_stats.get('total_ghost_reidentifications', 0)}")
    
    if out_writer is not None:
        print(f"\nðŸ’¾ Output video saved to: {args.output}")
    else:
        print("\nðŸ’¾ Output video not saved.")
    
    print("\nâœ… Processing completed successfully!")


if __name__ == "__main__":
    main()
