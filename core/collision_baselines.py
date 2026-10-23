"""
Calibration-light collision comparator baselines.

These detectors use the same tracked bounding-box metadata as TrackGuard-LTE,
but only apply one simple heuristic at a time. They are intended as paper
comparators, not as replacements for the Physics Primitive Layer.
"""

import math
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np


_MOTOR_VEHICLE_CLS = {
    "car", "van", "truck", "bus", "motor", "motorcycle",
    "tricycle", "awning-tricycle",
}


def _bbox_iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _diag(bbox) -> float:
    return math.hypot(max(bbox[2] - bbox[0], 1.0), max(bbox[3] - bbox[1], 1.0))


def _bbox_gap(a, b) -> float:
    """Shortest edge-to-edge distance between two axis-aligned boxes."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(math.hypot(dx, dy))


def _center(track) -> np.ndarray:
    c = getattr(track, "center", None)
    if c is None:
        c = track.current_detection.get("center", [0.0, 0.0])
    return np.asarray(c, dtype=float)


def _speed(track, back: int = 1) -> float:
    hist = getattr(track, "history", [])
    if len(hist) <= back:
        vel = getattr(track, "velocity", None)
        return float(np.linalg.norm(vel)) if vel is not None else 0.0
    c0 = np.asarray(hist[-1 - back].get("center", [0.0, 0.0]), dtype=float)
    c1 = np.asarray(hist[-1].get("center", [0.0, 0.0]), dtype=float)
    return float(np.linalg.norm(c1 - c0) / max(back, 1))


def _heading_change_deg(track, short_back: int = 2, long_back: int = 8) -> float:
    hist = getattr(track, "history", [])
    if len(hist) <= long_back:
        return 0.0
    c_now = np.asarray(hist[-1].get("center", [0.0, 0.0]), dtype=float)
    c_short = np.asarray(hist[-1 - short_back].get("center", [0.0, 0.0]), dtype=float)
    c_long = np.asarray(hist[-1 - long_back].get("center", [0.0, 0.0]), dtype=float)
    v1 = c_short - c_long
    v2 = c_now - c_short
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cosv)))


class TrackHeuristicCollisionBaseline:
    """
    Simple track-based collision baselines for comparison.

    Modes:
    - proximity: close centroids only.
    - iou-contact: bbox overlap only.
    - speed-drop-prox: speed drop with proximity.
    - heading-change: heading change with proximity.
    - union: any of the above.
    """

    detector_name = "heuristic_collision_baseline"

    def __init__(self, mode: str = "proximity", fps: float = 30.0):
        aliases = {
            "iou_contact": "iou-contact",
            "speed_drop_proximity": "speed-drop-prox",
            "heading_change_proximity": "heading-change",
        }
        self.mode = aliases.get(mode, mode)
        if self.mode not in {
            "proximity", "iou-contact", "speed-drop-prox", "heading-change", "union"
        }:
            raise ValueError(f"Unsupported collision baseline mode: {mode}")

        self.fps = float(fps)
        self.total_tracks_created_so_far = 0
        self.collision_pairs: Dict[Tuple[int, int], Dict] = {}
        self.proximity_warnings: Dict[Tuple[int, int], Tuple] = {}

        # Fixed comparator thresholds. Critical is the operator-review layer;
        # red is the stricter escalation layer.
        self.prox_critical = 3.0
        self.prox_red = 2.0
        self.iou_critical = 0.01
        self.iou_red = 0.05
        self.speed_drop_critical = 0.45
        self.speed_drop_red = 0.65
        self.heading_critical = 90.0
        self.heading_red = 135.0
        self.min_track_hits = 5

    def update_fps(self, fps: float) -> None:
        self.fps = float(fps)

    def update_scene_info(self, density: float, category: str) -> None:
        return None

    def detect(self, tracks: List, velocity_field=None) -> List[Dict]:
        active = [t for t in tracks if self._valid_track(t)]
        results = []
        current_pairs = set()

        for ti, tj in combinations(active, 2):
            pair_key = tuple(sorted((ti.track_id, tj.track_id)))
            evidence = self._pair_evidence(ti, tj)
            level = self._score_pair(evidence)
            if level is None:
                continue

            current_pairs.add(pair_key)
            state = self.collision_pairs.setdefault(pair_key, {"persist_count": 0})
            state["persist_count"] += 1
            state["state"] = "confirmed" if state["persist_count"] >= 1 else "monitoring"
            state["last_evidence"] = evidence

            results.append(self._make_detection(ti, tj, evidence, level, state["persist_count"]))

        # Keep only active comparator pairs to avoid stale suppression.
        self.collision_pairs = {
            k: v for k, v in self.collision_pairs.items()
            if k in current_pairs
        }
        return results

    def _valid_track(self, track) -> bool:
        if getattr(track, "state", "active") != "active":
            return False
        if getattr(track, "hits", 1) < self.min_track_hits:
            return False
        class_name = str(track.current_detection.get("class_name", "unknown")).lower()
        return class_name in _MOTOR_VEHICLE_CLS or class_name == "unknown"

    def _pair_evidence(self, ti, tj) -> Dict:
        bi = ti.current_detection.get("bbox", [0, 0, 1, 1])
        bj = tj.current_detection.get("bbox", [0, 0, 1, 1])
        ci = _center(ti)
        cj = _center(tj)
        mean_diag = max((_diag(bi) + _diag(bj)) / 2.0, 1.0)
        dist = float(np.linalg.norm(ci - cj))
        gap = _bbox_gap(bi, bj)
        center_ratio = dist / mean_diag
        gap_ratio = gap / mean_diag
        iou = _bbox_iou(bi, bj)

        si_now = _speed(ti, 1)
        sj_now = _speed(tj, 1)
        si_prev = _speed(ti, 5)
        sj_prev = _speed(tj, 5)
        drop_i = max(0.0, (si_prev - si_now) / max(si_prev, 1e-6)) if si_prev > 1.0 else 0.0
        drop_j = max(0.0, (sj_prev - sj_now) / max(sj_prev, 1e-6)) if sj_prev > 1.0 else 0.0
        max_speed_drop = max(drop_i, drop_j)

        heading_i = _heading_change_deg(ti)
        heading_j = _heading_change_deg(tj)
        max_heading_change = max(heading_i, heading_j)

        return {
            "bbox_i": bi,
            "bbox_j": bj,
            "center_dist": dist,
            "bbox_gap": gap,
            "prox_ratio": center_ratio,
            "gap_ratio": gap_ratio,
            "iou": iou,
            "speed_drop": max_speed_drop,
            "heading_change_deg": max_heading_change,
        }

    def _score_pair(self, ev: Dict) -> str:
        prox_crit = ev["prox_ratio"] <= self.prox_critical
        prox_red = ev["prox_ratio"] <= self.prox_red
        iou_crit = ev["iou"] >= self.iou_critical
        iou_red = ev["iou"] >= self.iou_red
        speed_crit = ev["speed_drop"] >= self.speed_drop_critical and prox_crit
        speed_red = ev["speed_drop"] >= self.speed_drop_red and prox_red
        heading_crit = ev["heading_change_deg"] >= self.heading_critical and prox_crit
        heading_red = ev["heading_change_deg"] >= self.heading_red and prox_red

        checks = {
            "proximity": (prox_crit, prox_red),
            "iou-contact": (iou_crit, iou_red),
            "speed-drop-prox": (speed_crit, speed_red),
            "heading-change": (heading_crit, heading_red),
            "union": (
                prox_crit or iou_crit or speed_crit or heading_crit,
                prox_red or iou_red or speed_red or heading_red,
            ),
        }
        critical, red = checks[self.mode]
        if red:
            return "red"
        if critical:
            return "critical"
        return None

    def _make_detection(self, ti, tj, ev: Dict, level: str, persist_count: int) -> Dict:
        bi = ev["bbox_i"]
        bj = ev["bbox_j"]
        union_bbox = [
            min(bi[0], bj[0]), min(bi[1], bj[1]),
            max(bi[2], bj[2]), max(bi[3], bj[3]),
        ]
        collision_point = [
            int((max(bi[0], bj[0]) + min(bi[2], bj[2])) / 2),
            int((max(bi[1], bj[1]) + min(bi[3], bj[3])) / 2),
        ]

        if level == "red":
            confidence = 70.0
            alert_level = "emergency"
            severity = "critical"
            label = "COLLISION DETECTED"
        else:
            confidence = 55.0
            alert_level = "warning"
            severity = "medium"
            label = "CRITICAL INTERACTION"

        return {
            "track_id": ti.track_id,
            "track_id_secondary": tj.track_id,
            "frame_id": getattr(ti, "current_frame", 0),
            "detector": self.detector_name,
            "behaviour_type": "collision",
            "bbox": union_bbox,
            "bbox_primary": bi,
            "bbox_secondary": bj,
            "class_primary": ti.current_detection.get("class_name", "unknown"),
            "class_secondary": tj.current_detection.get("class_name", "unknown"),
            "collision_point": collision_point,
            "iou_overlap": float(ev["iou"]),
            "energy_loss_primary": float("nan"),
            "energy_loss_secondary": float("nan"),
            "ars_zscore_i": float("nan"),
            "ars_zscore_j": float("nan"),
            "severity": severity,
            "alert_level": alert_level,
            "confidence": confidence,
            "confidence_label": label,
            "metric_value": float(ev["prox_ratio"]),
            "persistence": 1.0,
            "prediction_mode": "baseline",
            "state": "confirmed",
            "persist_count": persist_count,
            "detection_mode": f"baseline_{self.mode}",
            "tier": "baseline",
            "baseline_mode": self.mode,
            "baseline_center_dist": float(ev["center_dist"]),
            "baseline_bbox_gap": float(ev["bbox_gap"]),
            "baseline_prox_ratio": float(ev["prox_ratio"]),
            "baseline_gap_ratio": float(ev["gap_ratio"]),
            "baseline_speed_drop": float(ev["speed_drop"]),
            "baseline_heading_change_deg": float(ev["heading_change_deg"]),
        }
