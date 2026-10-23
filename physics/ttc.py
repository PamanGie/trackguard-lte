"""
Pixel-domain time-to-collision (TTC) baseline.

This module is intentionally independent from the proposed collision detector.
It provides a lightweight baseline signal for paper comparison only.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


VEHICLE_CLASSES = {"car", "truck", "bus", "van", "motorcycle", "motorbike"}


@dataclass(frozen=True)
class TTCConfig:
    warning_ttc_frames: float = 30.0
    confirmed_ttc_frames: float = 15.0
    proximity_multiplier: float = 2.0
    min_track_hits: int = 5
    min_closing_speed: float = 0.5


def _bbox_diag(track) -> float:
    bbox = getattr(track, "bbox", None) or getattr(track, "current_detection", {}).get("bbox")
    if not bbox or len(bbox) < 4:
        return 1.0
    x1, y1, x2, y2 = bbox[:4]
    return float(np.hypot(x2 - x1, y2 - y1))


def _track_class(track) -> str:
    return str(getattr(track, "current_detection", {}).get("class_name", "")).lower()


def _is_vehicle(track) -> bool:
    cls = _track_class(track)
    return not cls or cls in VEHICLE_CLASSES


def _center(track) -> np.ndarray:
    return np.asarray(getattr(track, "center", None) or getattr(track, "current_detection", {}).get("center"), dtype=float)


def _velocity(track) -> np.ndarray:
    return np.asarray(getattr(track, "velocity", np.zeros(2)), dtype=float)


def _confidence_from_ttc(ttc_frames: float, config: TTCConfig) -> float:
    if ttc_frames <= config.confirmed_ttc_frames:
        frac = 1.0 - max(ttc_frames, 0.0) / max(config.confirmed_ttc_frames, 1e-6)
        return 70.0 + 30.0 * frac
    if ttc_frames <= config.warning_ttc_frames:
        span = config.warning_ttc_frames - config.confirmed_ttc_frames
        frac = (config.warning_ttc_frames - ttc_frames) / max(span, 1e-6)
        return 50.0 + 20.0 * frac
    return 0.0


def evaluate_pair(track_i, track_j, frame_id: int, config: TTCConfig | None = None) -> dict | None:
    config = config or TTCConfig()
    if getattr(track_i, "hits", 0) < config.min_track_hits or getattr(track_j, "hits", 0) < config.min_track_hits:
        return None
    if not (_is_vehicle(track_i) and _is_vehicle(track_j)):
        return None

    ci = _center(track_i)
    cj = _center(track_j)
    vi = _velocity(track_i)
    vj = _velocity(track_j)
    if ci.shape[0] < 2 or cj.shape[0] < 2 or vi.shape[0] < 2 or vj.shape[0] < 2:
        return None

    rel_pos = cj[:2] - ci[:2]
    rel_vel = vj[:2] - vi[:2]
    rel_speed_sq = float(np.dot(rel_vel, rel_vel))
    if rel_speed_sq <= 1e-6:
        return None

    ttc_frames = -float(np.dot(rel_pos, rel_vel)) / rel_speed_sq
    if ttc_frames <= 0 or ttc_frames > config.warning_ttc_frames:
        return None

    closest = rel_pos + rel_vel * ttc_frames
    closest_dist = float(np.linalg.norm(closest))
    scale = max(_bbox_diag(track_i), _bbox_diag(track_j), 1.0)
    if closest_dist > config.proximity_multiplier * scale:
        return None

    current_dist = float(np.linalg.norm(rel_pos))
    closing_speed = -float(np.dot(rel_pos, rel_vel)) / max(current_dist, 1e-6)
    if closing_speed < config.min_closing_speed:
        return None

    confidence = _confidence_from_ttc(ttc_frames, config)
    if confidence < 50.0:
        return None

    level = "emergency" if confidence >= 70.0 else "warning"
    return {
        "method": "ttc",
        "frame": frame_id,
        "track_id": getattr(track_i, "track_id", -1),
        "track_id_secondary": getattr(track_j, "track_id", -1),
        "confidence": float(min(100.0, max(0.0, confidence))),
        "alert_level": level,
        "raw_value": float(ttc_frames),
        "closest_distance": closest_dist,
        "distance": current_dist,
        "closing_speed": closing_speed,
    }


def detect_ttc(tracks: Iterable, frame_id: int, config: TTCConfig | None = None) -> list[dict]:
    config = config or TTCConfig()
    detections = []
    active_tracks = list(tracks)
    for track_i, track_j in combinations(active_tracks, 2):
        det = evaluate_pair(track_i, track_j, frame_id, config)
        if det is not None:
            detections.append(det)
    return detections
