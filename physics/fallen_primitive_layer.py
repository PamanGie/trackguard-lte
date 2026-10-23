"""
Fallen Primitive Layer for LTE-TrackGuard
=========================================

Single-track Physics Primitive Layer (PPL) for motorcycle fall detection.

This layer is intentionally tracker-agnostic. It consumes only track metadata:
track_id, history entries with center/bbox, current_detection, hits, and
current_frame. It does not replace the legacy FallenDetector heuristics; it
provides a physics-evidence path and scalar logging for MONET development.
"""

import csv
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class FallenPrimitiveLayer:
    """
    Calibration-light single-track primitives for motorcycle fall evidence.

    Initial primitive set:
      PCS: Posture Collapse Signal, based on self-normalized aspect-ratio shock.
      KCR: Kinetic Collapse Ratio, a single-track speed-loss analogue of KELR.
      PPS: Posture Persistence Signal, persistence of collapsed posture.
      SDE: Sliding Dissipation Evidence, post-collapse monotonic speed decay.
      RCG: Rider Coupling Gap, optional person-motorcycle separation cue.
      Corner guard: suppresses controlled cornering / aggressive lean.
    """

    def __init__(self, fps: float = 30.0, config: Optional[Dict] = None):
        config = config or {}
        self.fps = float(config.get("fps", fps))
        self.enabled = bool(config.get("enabled", True))
        self.enable_legacy_fallback = bool(config.get("enable_legacy_fallback", True))

        self.W = max(8, int(self.fps * float(config.get("window_seconds", 1.0))))
        self.after_n = max(3, int(self.fps * float(config.get("after_seconds", 0.12))))
        self.persistence_n = max(3, int(self.fps * float(config.get("persistence_seconds", 0.4))))
        self.slide_n = max(3, int(self.fps * float(config.get("slide_seconds", 0.35))))
        self.f2_sw_n = max(6, int(self.fps * float(config.get("f2_sw_seconds", 0.67))))

        self.pcs_z_threshold = float(config.get("pcs_z_threshold", 2.0))
        self.pcs_quantile = float(config.get("pcs_quantile", 0.30))
        self.min_ar_drop = float(config.get("min_ar_drop", 0.15))

        self.kcr_threshold = float(config.get("kcr_threshold", 0.70))
        self.kcr_z_threshold = float(config.get("kcr_z_threshold", 2.0))
        self.min_before_speed = float(config.get("min_before_speed", 1.0))

        self.pps_threshold = float(config.get("pps_threshold", 0.60))
        self.sde_threshold = float(config.get("sde_threshold", 0.55))
        self.sde_min_step_drop = float(config.get("sde_min_step_drop", 0.03))
        self.sde_total_drop = float(config.get("sde_total_drop", 0.25))
        self.f2d_ar_drop_threshold = float(config.get("f2d_ar_drop_threshold", 0.35))
        self.f2d_pps_threshold = float(config.get("f2d_pps_threshold", 0.70))
        self.f2d_sde_threshold = float(config.get("f2d_sde_threshold", 0.55))

        self.corner_mobility_threshold = float(config.get("corner_mobility_threshold", 0.75))
        self.corner_heading_threshold = float(config.get("corner_heading_threshold", 0.70))
        self.corner_min_speed = float(config.get("corner_min_speed", 2.0))

        self.rider_near_threshold = float(config.get("rider_near_threshold", 1.2))
        self.rider_sep_threshold = float(config.get("rider_sep_threshold", 0.8))
        self.rider_lookback = max(3, int(self.fps * float(config.get("rider_lookback_seconds", 0.5))))

        self._scalar_log_enabled = bool(config.get("scalar_logging_enabled", False))
        self._scalar_log: List[Dict] = []
        self._scalar_log_video = ""
        self._scalar_log_label = 0
        self._last_scalar_by_track: Dict[int, Dict] = {}
        self._primitive_history: Dict[int, List[Dict]] = {}

        print("  Fallen PPL: ENABLED" if self.enabled else "  Fallen PPL: disabled")
        print(
            f"    W={self.W}, after={self.after_n}, persist={self.persistence_n}, "
            f"KCR>{self.kcr_threshold:.2f}, PCS z>{self.pcs_z_threshold:.1f}"
        )

    def update_fps(self, fps: float) -> None:
        """Update temporal windows when video FPS is known."""
        self.fps = float(fps)
        self.W = max(8, int(self.fps * 1.0))
        self.after_n = max(3, int(self.fps * 0.12))
        self.persistence_n = max(3, int(self.fps * 0.4))
        self.slide_n = max(3, int(self.fps * 0.35))
        self.f2_sw_n = max(6, int(self.fps * 0.67))
        self.rider_lookback = max(3, int(self.fps * 0.5))

    def reset(self) -> None:
        """Reset per-video scalar history."""
        self._scalar_log.clear()
        self._primitive_history.clear()
        self._last_scalar_by_track.clear()

    def enable_scalar_logging(self, video_name: str = "", label: int = 0) -> None:
        self._scalar_log_enabled = True
        self._scalar_log_video = video_name
        self._scalar_log_label = int(label)
        self._scalar_log.clear()

    def disable_scalar_logging(self) -> None:
        self._scalar_log_enabled = False

    def get_scalar_log(self) -> List[Dict]:
        return list(self._scalar_log)

    def get_last_scalar(self, track_id: int) -> Dict:
        """Return the latest scalar probe for a track, even when PPL did not fire."""
        return dict(self._last_scalar_by_track.get(int(track_id), {}))

    def clear_scalar_log(self) -> None:
        self._scalar_log.clear()

    def save_scalar_log_csv(self, path: str) -> None:
        if not self._scalar_log:
            return
        keys = list(self._scalar_log[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._scalar_log)

    def check_track(
        self,
        track,
        frame_id: int,
        nearby_tracks: Optional[List] = None,
    ) -> Optional[Dict]:
        """
        Evaluate one motorcycle track for fall evidence.

        Returns a dict when a PPL path triggers, otherwise None. Scalar values
        are logged for every valid motorcycle track when scalar logging is on.
        """
        if not self.enabled:
            return None
        if not self._is_valid_motorcycle(track):
            return None

        pcs, pcs_data = self._compute_pcs(track)
        kcr, kcr_data = self._compute_kcr(track)
        pps, pps_data = self._compute_pps(track)
        sde, sde_data = self._compute_sde(track)
        rcg, rcg_data = self._compute_rcg(track, nearby_tracks or [])
        corner_guard, corner_data = self._compute_corner_guard(track, pcs, kcr, pps, kcr_data)
        track_id = int(getattr(track, "track_id", -1))
        sw_data = self._update_temporal_primitives(
            track_id=track_id,
            frame_id=int(frame_id),
            pcs=pcs,
            pps=pps,
            sde=sde,
            corner_guard=corner_guard,
        )

        path_f1 = pcs and kcr and pps and not corner_guard
        path_f2 = pcs and sde and pps and not corner_guard
        path_f2_sw = sw_data["f2_sw"] and not path_f2 and not path_f1
        path_f2d = self._compute_drop_slide_path(
            pcs=pcs,
            pps=pps,
            sde=sde,
            corner_guard=corner_guard,
            pcs_data=pcs_data,
            pps_data=pps_data,
            sde_data=sde_data,
        ) and not path_f1 and not path_f2 and not path_f2_sw
        path_f3 = pcs and rcg and (kcr or sde or pps)

        triggered = path_f1 or path_f2 or path_f2_sw or path_f2d or path_f3
        path = "none"
        if path_f1:
            path = "F1_POSTURE_ENERGY"
        elif path_f2:
            path = "F2_POSTURE_SLIDE"
        elif path_f2_sw:
            path = "F2_SW_POSTURE_SLIDE"
        elif path_f2d:
            path = "F2D_DROP_SLIDE"
        elif path_f3:
            path = "F3_RIDER_SEPARATION"

        primitives = []
        effective_pcs = pcs or path_f2_sw or path_f2d
        effective_pps = pps or path_f2_sw
        effective_sde = sde or path_f2_sw
        if path_f2d and not pcs:
            primitives.append("PCS_DROP")
        elif effective_pcs:
            primitives.append("PCS")
        if kcr:
            primitives.append("KCR")
        if effective_pps:
            primitives.append("PPS")
        if effective_sde:
            primitives.append("SDE")
        if rcg:
            primitives.append("RCG")
        if corner_guard:
            primitives.append("CORNER_GUARD")

        scalar = {
            "video": self._scalar_log_video,
            "label": self._scalar_log_label,
            "frame_id": int(frame_id),
            "track_id": track_id,
            "class_name": track.current_detection.get("class_name", "unknown"),
            "triggered": int(triggered),
            "path": path,
            "pcs": int(pcs),
            "kcr": int(kcr),
            "pps": int(pps),
            "sde": int(sde),
            "rcg": int(rcg),
            "corner_guard": int(corner_guard),
            "f2d_drop_slide": int(path_f2d),
            "f2d_ar_change": float(abs(pcs_data.get("ar_drop", 0.0))),
            **pcs_data,
            **kcr_data,
            **pps_data,
            **sde_data,
            **rcg_data,
            **corner_data,
            **sw_data,
        }
        if self._scalar_log_enabled:
            self._scalar_log.append(scalar)
        self._last_scalar_by_track[int(getattr(track, "track_id", -1))] = scalar

        if not triggered:
            return None

        confidence = self._confidence_from_primitives(
            effective_pcs, kcr, effective_pps, effective_sde, rcg, corner_guard
        )
        return {
            "frame_id": int(frame_id),
            "track_id": track_id,
            "path_triggered": path,
            "primitives_fired": primitives,
            "confidence": float(confidence),
            "pcs": bool(effective_pcs),
            "kcr": bool(kcr),
            "pps": bool(effective_pps),
            "sde": bool(effective_sde),
            "rcg": bool(rcg),
            "corner_guard": bool(corner_guard),
            "scalar": scalar,
        }

    def _compute_drop_slide_path(
        self,
        pcs: bool,
        pps: bool,
        sde: bool,
        corner_guard: bool,
        pcs_data: Dict,
        pps_data: Dict,
        sde_data: Dict,
    ) -> bool:
        """
        Drop-supported F2 variant for cases where aspect-ratio change is large,
        but the strict PCS z/quantile gate misses the fall posture.
        """
        if pcs or corner_guard:
            return False
        ar_change = abs(float(pcs_data.get("ar_drop", 0.0)))
        pps_score = float(pps_data.get("pps_score", 0.0))
        sde_score = float(sde_data.get("sde_score", 0.0))
        return (
            ar_change >= self.f2d_ar_drop_threshold and
            pps and
            sde and
            pps_score >= self.f2d_pps_threshold and
            sde_score >= self.f2d_sde_threshold
        )

    def _update_temporal_primitives(
        self,
        track_id: int,
        frame_id: int,
        pcs: bool,
        pps: bool,
        sde: bool,
        corner_guard: bool,
    ) -> Dict:
        """
        Track short-window primitive co-occurrence for F2-sw.

        F2-sw keeps the same physical evidence as F2 (PCS+PPS+SDE), but allows
        the three primitives to arrive within a bounded temporal window instead
        of requiring exact frame-level coincidence.
        """
        if track_id < 0:
            return self._f2_sw_defaults()

        entries = self._primitive_history.setdefault(track_id, [])
        entries.append({
            "frame_id": int(frame_id),
            "pcs": bool(pcs),
            "pps": bool(pps),
            "sde": bool(sde),
            "corner_guard": bool(corner_guard),
        })
        min_frame = int(frame_id) - self.f2_sw_n
        entries[:] = [entry for entry in entries if entry["frame_id"] >= min_frame]

        pcs_frames = [entry["frame_id"] for entry in entries if entry["pcs"]]
        pps_frames = [entry["frame_id"] for entry in entries if entry["pps"]]
        sde_frames = [entry["frame_id"] for entry in entries if entry["sde"]]
        corner_hits = sum(1 for entry in entries if entry["corner_guard"])

        if pcs_frames and pps_frames and sde_frames:
            involved = [pcs_frames[-1], pps_frames[-1], sde_frames[-1]]
            span = max(involved) - min(involved)
            f2_sw = span <= self.f2_sw_n and corner_hits == 0
        else:
            span = 0
            f2_sw = False

        return {
            "f2_sw": int(f2_sw),
            "f2_sw_window": int(self.f2_sw_n),
            "f2_sw_span": int(span),
            "f2_sw_pcs_hits": int(len(pcs_frames)),
            "f2_sw_pps_hits": int(len(pps_frames)),
            "f2_sw_sde_hits": int(len(sde_frames)),
            "f2_sw_corner_hits": int(corner_hits),
        }

    def _f2_sw_defaults(self) -> Dict:
        return {
            "f2_sw": 0,
            "f2_sw_window": int(self.f2_sw_n),
            "f2_sw_span": 0,
            "f2_sw_pcs_hits": 0,
            "f2_sw_pps_hits": 0,
            "f2_sw_sde_hits": 0,
            "f2_sw_corner_hits": 0,
        }

    def _compute_pcs(self, track) -> Tuple[bool, Dict]:
        ar_values = self._aspect_ratios(track, last_n=self.W + self.persistence_n + 1)
        if len(ar_values) < 5:
            return False, self._pcs_defaults()

        if len(ar_values) > self.persistence_n + 5:
            baseline = ar_values[:-self.persistence_n]
        else:
            baseline = ar_values[:-1]
        ar_now = ar_values[-1]
        mean_ar = float(np.mean(baseline))
        std_ar = float(np.std(baseline))
        q_ar = float(np.quantile(baseline, self.pcs_quantile))
        z_ar = abs(ar_now - mean_ar) / (std_ar + 1e-6)
        ar_drop = (mean_ar - ar_now) / max(mean_ar, 1e-6)

        fired = (
            z_ar > self.pcs_z_threshold and
            ar_now < q_ar and
            ar_drop > self.min_ar_drop
        )
        return fired, {
            "ar_now": float(ar_now),
            "ar_mean": float(mean_ar),
            "ar_std": float(std_ar),
            "ar_q_low": float(q_ar),
            "ar_z": float(z_ar),
            "ar_drop": float(ar_drop),
        }

    def _compute_kcr(self, track) -> Tuple[bool, Dict]:
        speeds = self._speeds(track, last_n=self.W + 1)
        if len(speeds) < self.after_n + 3:
            return False, self._kcr_defaults()

        n_after = min(self.after_n, max(3, len(speeds) // 3))
        speeds_before = speeds[:-n_after]
        speeds_after = speeds[-n_after:]
        mean_before = float(np.mean(speeds_before))
        mean_after = float(np.mean(speeds_after))
        std_before = float(np.std(speeds_before))

        if mean_before < self.min_before_speed:
            return False, {
                "kcr_value": 0.0,
                "kcr_z": 0.0,
                "speed_before": mean_before,
                "speed_after": mean_after,
            }

        kcr_value = 1.0 - (mean_after ** 2) / (mean_before ** 2 + 1e-6)
        kcr_z = (mean_before - mean_after) / (std_before + 1e-6)
        fired = kcr_value > self.kcr_threshold and kcr_z > self.kcr_z_threshold
        return fired, {
            "kcr_value": float(kcr_value),
            "kcr_z": float(kcr_z),
            "speed_before": mean_before,
            "speed_after": mean_after,
        }

    def _compute_pps(self, track) -> Tuple[bool, Dict]:
        ar_values = self._aspect_ratios(track, last_n=max(self.W + 1, self.persistence_n + 6))
        if len(ar_values) < self.persistence_n + 4:
            return False, {"pps_score": 0.0}

        baseline = ar_values[: -self.persistence_n]
        recent = ar_values[-self.persistence_n :]
        if len(baseline) < 4:
            return False, {"pps_score": 0.0}

        mean_ar = float(np.mean(baseline))
        std_ar = float(np.std(baseline))
        q_ar = float(np.quantile(baseline, self.pcs_quantile))
        collapsed = []
        for ar in recent:
            z = abs(ar - mean_ar) / (std_ar + 1e-6)
            collapsed.append(1 if (ar < q_ar or z > self.pcs_z_threshold) else 0)
        pps_score = float(sum(collapsed) / len(collapsed))
        return pps_score > self.pps_threshold, {"pps_score": pps_score}

    def _compute_sde(self, track) -> Tuple[bool, Dict]:
        speeds = self._speeds(track, last_n=self.slide_n + 2)
        if len(speeds) < 4:
            return False, {"sde_score": 0.0, "sde_total_drop": 0.0}

        drops = 0
        comparisons = 0
        for idx in range(1, len(speeds)):
            comparisons += 1
            if speeds[idx] < speeds[idx - 1] * (1.0 - self.sde_min_step_drop):
                drops += 1
        sde_score = float(drops / max(comparisons, 1))
        start_speed = float(np.mean(speeds[:2]))
        end_speed = float(np.mean(speeds[-2:]))
        total_drop = (start_speed - end_speed) / max(start_speed, 1e-6)
        fired = sde_score > self.sde_threshold and total_drop > self.sde_total_drop
        return fired, {"sde_score": sde_score, "sde_total_drop": float(total_drop)}

    def _compute_rcg(self, track, nearby_tracks: List) -> Tuple[bool, Dict]:
        motor_center = self._current_center(track)
        motor_diag = self._bbox_diag(track.current_detection.get("bbox"))
        if motor_center is None or motor_diag <= 1e-6:
            return False, {"rcg_gap_now": 0.0, "rcg_gap_delta": 0.0}

        best_person = None
        best_dist = float("inf")
        for other in nearby_tracks:
            if other is track or not hasattr(other, "current_detection"):
                continue
            cls = (other.current_detection.get("class_name") or "").lower()
            if cls not in ("person", "pedestrian", "people"):
                continue
            center = self._current_center(other)
            if center is None:
                continue
            dist = float(np.linalg.norm(np.array(motor_center) - np.array(center)))
            if dist < best_dist:
                best_dist = dist
                best_person = other

        if best_person is None:
            return False, {"rcg_gap_now": 0.0, "rcg_gap_delta": 0.0}

        gap_now = best_dist / motor_diag
        motor_past = self._center_from_history(track, self.rider_lookback)
        person_past = self._center_from_history(best_person, self.rider_lookback)
        if motor_past is None or person_past is None:
            return False, {"rcg_gap_now": float(gap_now), "rcg_gap_delta": 0.0}

        gap_past = float(np.linalg.norm(np.array(motor_past) - np.array(person_past))) / motor_diag
        gap_delta = gap_now - gap_past
        fired = gap_past < self.rider_near_threshold and gap_delta > self.rider_sep_threshold
        return fired, {"rcg_gap_now": float(gap_now), "rcg_gap_delta": float(gap_delta)}

    def _compute_corner_guard(
        self,
        track,
        pcs: bool,
        kcr: bool,
        pps: bool,
        kcr_data: Dict,
    ) -> Tuple[bool, Dict]:
        if not pcs:
            return False, {"corner_mobility": 0.0, "corner_heading_cos": 0.0}

        centers = self._centers(track, last_n=self.W)
        if len(centers) < 3:
            return False, {"corner_mobility": 0.0, "corner_heading_cos": 0.0}

        net_vec = np.array(centers[-1]) - np.array(centers[0])
        inst_vec = np.array(centers[-1]) - np.array(centers[-2])
        net_norm = float(np.linalg.norm(net_vec))
        inst_norm = float(np.linalg.norm(inst_vec))
        diag = self._bbox_diag(track.current_detection.get("bbox"))
        mobility = net_norm / max(diag, 1e-6)
        heading_cos = 0.0
        if net_norm > 1e-6 and inst_norm > 1e-6:
            heading_cos = float(np.dot(net_vec, inst_vec) / (net_norm * inst_norm))

        speed_after = float(kcr_data.get("speed_after", 0.0))
        fired = (
            mobility > self.corner_mobility_threshold and
            heading_cos > self.corner_heading_threshold and
            speed_after > self.corner_min_speed and
            not kcr
        )
        return fired, {
            "corner_mobility": float(mobility),
            "corner_heading_cos": float(heading_cos),
        }

    def _confidence_from_primitives(
        self,
        pcs: bool,
        kcr: bool,
        pps: bool,
        sde: bool,
        rcg: bool,
        corner_guard: bool,
    ) -> float:
        score = 0.0
        score += 0.30 if pcs else 0.0
        score += 0.30 if kcr else 0.0
        score += 0.20 if pps else 0.0
        score += 0.15 if sde else 0.0
        score += 0.15 if rcg else 0.0
        score -= 0.40 if corner_guard else 0.0
        return max(0.0, min(1.0, score))

    def _is_valid_motorcycle(self, track) -> bool:
        if not hasattr(track, "current_detection"):
            return False
        if getattr(track, "hits", 0) < 3:
            return False
        if getattr(track, "state", "") == "ghost":
            return False
        if getattr(track, "misses", 0) > 0:
            return False
        cls = (track.current_detection.get("class_name") or "").lower()
        return cls == "motorcycle"

    def _aspect_ratios(self, track, last_n: int) -> List[float]:
        values = []
        for entry in self._entries_with_current(track, last_n):
            bbox = entry.get("bbox")
            if bbox is None:
                continue
            w = max(float(bbox[2] - bbox[0]), 1.0)
            h = max(float(bbox[3] - bbox[1]), 1.0)
            values.append(h / w)
        return values

    def _speeds(self, track, last_n: int) -> List[float]:
        centers = self._centers(track, last_n)
        if len(centers) < 3:
            return []
        speeds = []
        for idx in range(2, len(centers)):
            v = np.array(centers[idx]) - np.array(centers[idx - 2])
            speeds.append(float(np.linalg.norm(v)) / 2.0)
        return speeds

    def _centers(self, track, last_n: int) -> List:
        centers = []
        for entry in self._entries_with_current(track, last_n):
            center = entry.get("center")
            if center is not None:
                centers.append(center)
        return centers

    def _entries_with_current(self, track, last_n: int) -> List[Dict]:
        history = list(getattr(track, "history", []))
        current = getattr(track, "current_detection", None)
        if current is not None:
            if not history or history[-1] is not current:
                history.append(current)
        return history[-last_n:] if last_n > 0 else history

    def _current_center(self, track) -> Optional[List[float]]:
        det = getattr(track, "current_detection", {})
        center = det.get("center")
        if center is not None:
            return center
        bbox = det.get("bbox")
        if bbox is None:
            return None
        return [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]

    def _center_from_history(self, track, frames_back: int) -> Optional[List[float]]:
        entries = self._entries_with_current(track, frames_back + 1)
        if len(entries) < frames_back + 1:
            return None
        return entries[0].get("center")

    def _bbox_diag(self, bbox) -> float:
        if bbox is None:
            return 0.0
        return math.sqrt(max(float(bbox[2] - bbox[0]), 1.0) ** 2 +
                         max(float(bbox[3] - bbox[1]), 1.0) ** 2)

    def _pcs_defaults(self) -> Dict:
        return {
            "ar_now": 0.0,
            "ar_mean": 0.0,
            "ar_std": 0.0,
            "ar_q_low": 0.0,
            "ar_z": 0.0,
            "ar_drop": 0.0,
        }

    def _kcr_defaults(self) -> Dict:
        return {
            "kcr_value": 0.0,
            "kcr_z": 0.0,
            "speed_before": 0.0,
            "speed_after": 0.0,
        }
