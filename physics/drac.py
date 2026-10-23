"""
Pixel-domain deceleration rate to avoid a crash (DRAC) baseline.

This module is intentionally independent from the proposed collision detector.
It provides a lightweight baseline signal for paper comparison only.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from physics.ttc import VEHICLE_CLASSES, _bbox_diag, _center, _track_class, _velocity


@dataclass(frozen=True)
class DRACConfig:
    warning_drac: float = 0.03
    confirmed_drac: float = 0.07
    proximity_multiplier: float = 2.5
    min_track_hits: int = 5
    min_closing_speed: float = 0.5


def _is_vehicle(track) -> bool:
    cls = _track_class(track)
    return not cls or cls in VEHICLE_CLASSES


def _confidence_from_drac(drac: float, config: DRACConfig) -> float:
    if drac >= config.confirmed_drac:
        scale = max(config.confirmed_drac, 1e-6)
        return min(100.0, 70.0 + 30.0 * ((drac - config.confirmed_drac) / scale))
    if drac >= config.warning_drac:
        span = config.confirmed_drac - config.warning_drac
        frac = (drac - config.warning_drac) / max(span, 1e-6)
        return 50.0 + 20.0 * frac
    return 0.0


def evaluate_pair(track_i, track_j, frame_id: int, config: DRACConfig | None = None) -> dict | None:
    config = config or DRACConfig()
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
    distance = float(np.linalg.norm(rel_pos))
    if distance <= 1e-6:
        return None

    rel_vel = vj[:2] - vi[:2]
    unit = rel_pos / distance
    closing_speed = -float(np.dot(rel_vel, unit))
    if closing_speed < config.min_closing_speed:
        return None

    scale = max(_bbox_diag(track_i), _bbox_diag(track_j), 1.0)
    center_gap = max(distance - 0.5 * (_bbox_diag(track_i) + _bbox_diag(track_j)), 1.0)
    if center_gap > config.proximity_multiplier * scale:
        return None

    drac = (closing_speed * closing_speed) / (2.0 * center_gap)
    confidence = _confidence_from_drac(drac, config)
    if confidence < 50.0:
        return None

    level = "emergency" if confidence >= 70.0 else "warning"
    return {
        "method": "drac",
        "frame": frame_id,
        "track_id": getattr(track_i, "track_id", -1),
        "track_id_secondary": getattr(track_j, "track_id", -1),
        "confidence": float(min(100.0, max(0.0, confidence))),
        "alert_level": level,
        "raw_value": float(drac),
        "distance": distance,
        "gap": center_gap,
        "closing_speed": closing_speed,
    }


def detect_drac(tracks: Iterable, frame_id: int, config: DRACConfig | None = None) -> list[dict]:
    config = config or DRACConfig()
    detections = []
    active_tracks = list(tracks)
    for track_i, track_j in combinations(active_tracks, 2):
        det = evaluate_pair(track_i, track_j, frame_id, config)
        if det is not None:
            detections.append(det)
    return detections
