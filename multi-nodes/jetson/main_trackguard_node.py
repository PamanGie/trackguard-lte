"""
TrackGuard-LTE Jetson node.

Runs the repository-level TrackGuard collision pipeline and publishes compact
alarm/resource events to MQTT for multi-node IoT experiments.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "main.py"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config_module = _load_module("trackguard_jetson_config", Path(__file__).parent / "config.py")
JetsonConfig = config_module.JetsonConfig
publisher_module = _load_module("trackguard_mqtt_publisher", Path(__file__).parent / "mqtt_publisher.py")
MQTTPublisher = publisher_module.MQTTPublisher
resource_module = _load_module("trackguard_resource_monitor", Path(__file__).parent / "resource_monitor.py")
ResourceMonitor = resource_module.ResourceMonitor


ALL_DET_RE = re.compile(
    r"ALL_DETECTION\s+frame=(?P<frame>\d+)\s+level=(?P<level>\w+)\s+"
    r"(?:conf=(?P<conf>[-\d.]+)\s+)?(?:label=(?P<label>[^\s]+)\s+)?"
    r"ki=(?P<ki>[^\s]+)\s+kj=(?P<kj>[^\s]+)\s+"
    r"azi=(?P<azi>[^\s]+)\s+azj=(?P<azj>[^\s]+)\s+"
    r"iou=(?P<iou>[^\s]+)\s+mode=(?P<mode>[^\s]+)\s+"
    r"i=(?P<bbox_i>[^\s]+)\s+j=(?P<bbox_j>[^\s]+)"
)
EVIDENCE_RE = re.compile(r"\[EVIDENCE SNAPSHOT\].*?:\s+(?P<path>.+)$")
FRAMES_RE = re.compile(r"Frames Processed:\s+(?P<frames>\d+)")
FPS_RE = re.compile(r"Average FPS:\s+(?P<fps>[-\d.]+)")
MS_RE = re.compile(r"Average Processing Time:\s+(?P<ms>[-\d.]+)\s+ms/frame")


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value in {"nan", "None"}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def parse_bbox(value: str) -> Optional[list[float]]:
    if value == "None":
        return None
    try:
        return [float(v) for v in value.split(",")]
    except Exception:
        return None


def is_url_or_camera(source: str) -> bool:
    return source.strip().isdigit() or "://" in source


def safe_stem(value: str) -> str:
    stem = Path(value).stem if "://" not in value else value.replace("://", "_")
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)[:80] or "source"


def list_video_files(source: str) -> list[Path]:
    path = Path(source)
    if not path.is_dir():
        return []
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(files, key=lambda p: p.name.lower())


def run_folder_mode(args: argparse.Namespace, videos: list[Path]) -> int:
    print("=" * 72)
    print("TrackGuard-LTE Folder Mode")
    print("=" * 72)
    print(f"Node/Jetson : {args.jetson_id}")
    print(f"Camera ID   : {args.camera_id}")
    print(f"Folder      : {args.source}")
    print(f"Videos      : {len(videos)}")
    print(f"MQTT broker : {args.mqtt_host}:{args.mqtt_port}")
    print("=" * 72)

    script_path = Path(__file__).resolve()
    common = [
        sys.executable,
        str(script_path),
        "--jetson-id", args.jetson_id,
        "--camera-id", args.camera_id,
        "--mqtt-host", args.mqtt_host,
        "--mqtt-port", str(args.mqtt_port),
        "--model", args.model,
        "--tracker", args.tracker,
        "--conf", str(args.conf),
        "--camera-angle", args.camera_angle,
        "--device-mode", args.device_mode,
        "--heartbeat-interval", str(args.heartbeat_interval),
        "--reporting-mode", args.reporting_mode,
        "--episode-gap-seconds", str(args.episode_gap_seconds),
    ]
    if args.max_frames:
        common += ["--max-frames", str(args.max_frames)]
    if args.fps_limit:
        common += ["--fps-limit", str(args.fps_limit)]
    if args.live_refresh_interval is not None:
        common += ["--live-refresh-interval", str(args.live_refresh_interval)]
    if args.evidence_dir:
        common += ["--evidence-dir", args.evidence_dir]
    if args.no_evidence_snapshots:
        common += ["--no-evidence-snapshots"]
    if args.stage_profile_dir:
        common += ["--stage-profile-dir", args.stage_profile_dir]

    failed = 0
    start = time.time()
    for idx, video in enumerate(videos, 1):
        print("\n" + "=" * 72)
        print(f"[{idx}/{len(videos)}] {video.name}")
        print("=" * 72)
        cmd = common + ["--source", str(video)]
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            failed += 1
            print(f"WARNING: {video.name} exited with code {result.returncode}")

    elapsed = time.time() - start
    print("\n" + "=" * 72)
    print("TrackGuard folder summary")
    print("=" * 72)
    print(f"Videos processed : {len(videos)}")
    print(f"Failed videos    : {failed}")
    print(f"Runtime          : {elapsed:.2f} s")
    print("=" * 72)
    return 1 if failed else 0


def build_trackguard_command(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(MAIN_PY)]
    if is_url_or_camera(args.source):
        cmd += ["--source", args.source]
    else:
        cmd += ["--video", args.source]

    cmd += [
        "--physics",
        "--detect", "collision",
        "--tracker", args.tracker,
        "--model", args.model,
        "--conf", str(args.conf),
        "--camera-angle", args.camera_angle,
        "--no-output",
        "--no-display",
    ]

    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]
    if args.fps_limit:
        cmd += ["--fps-limit", str(args.fps_limit)]
    if args.live_refresh_interval is not None:
        cmd += ["--live-refresh-interval", str(args.live_refresh_interval)]
    if args.evidence_dir:
        cmd += ["--evidence-dir", args.evidence_dir]
    if args.no_evidence_snapshots:
        cmd += ["--no-evidence-snapshots"]
    if getattr(args, "stage_profile_output", None):
        cmd += ["--stage-profile-output", args.stage_profile_output]

    return cmd


def make_event_message(
    args: argparse.Namespace,
    match: re.Match,
    resource_snapshot: Dict[str, float],
    event_index: int,
    start_time: float,
) -> Dict[str, Any]:
    detected_time = time.time()
    level = match.group("level").lower()
    confidence = parse_float(match.group("conf"))
    frame_id = int(match.group("frame"))
    elapsed = max(detected_time - start_time, 1e-6)

    message = {
        "schema_version": "trackguard-iot-v1",
        "project": "TrackGuard-LTE",
        "message_type": "alarm",
        "jetson_id": args.jetson_id,
        "camera_id": args.camera_id,
        "source": args.source,
        "frame_id": frame_id,
        "event_index": event_index,
        "alarm_level": level,
        "severity_index": confidence,
        "label": match.group("label") or "",
        "mode": match.group("mode"),
        "streaming_mode": "event-based",
        "device_mode": args.device_mode,
        "timestamps": {
            "capture_time": detected_time,
            "detect_time": detected_time,
            "send_time": time.time(),
            "node_start_time": start_time,
        },
        "evidence": {
            "kelr_i": parse_float(match.group("ki")),
            "kelr_j": parse_float(match.group("kj")),
            "ars_zi": parse_float(match.group("azi")),
            "ars_zj": parse_float(match.group("azj")),
            "iou": parse_float(match.group("iou")),
            "bbox_i": parse_bbox(match.group("bbox_i")),
            "bbox_j": parse_bbox(match.group("bbox_j")),
        },
        "node_metrics": {
            "runtime_s": elapsed,
            "event_rate_per_min": (event_index / elapsed) * 60.0,
            "resource": resource_snapshot,
        },
        "summary": {
            "num_detections": 1,
            "num_confirmed": 1 if level == "emergency" else 0,
            "num_monitoring": 1 if level in {"warning", "caution"} else 0,
            "red_escalation": level == "emergency",
        },
    }
    return message


def make_status_message(
    args: argparse.Namespace,
    resource_snapshot: Dict[str, float],
    start_time: float,
    frames_processed: Optional[int] = None,
    avg_fps: Optional[float] = None,
    avg_ms_frame: Optional[float] = None,
) -> Dict[str, Any]:
    now = time.time()
    return {
        "schema_version": "trackguard-iot-v1",
        "project": "TrackGuard-LTE",
        "message_type": "status",
        "jetson_id": args.jetson_id,
        "camera_id": args.camera_id,
        "source": args.source,
        "frame_id": frames_processed or 0,
        "streaming_mode": "event-based",
        "device_mode": args.device_mode,
        "timestamps": {
            "capture_time": now,
            "detect_time": now,
            "send_time": now,
            "node_start_time": start_time,
        },
        "node_metrics": {
            "runtime_s": now - start_time,
            "frames_processed": frames_processed,
            "avg_fps": avg_fps,
            "avg_ms_frame": avg_ms_frame,
            "resource": resource_snapshot,
        },
        "summary": {
            "num_detections": 0,
            "num_confirmed": 0,
            "num_monitoring": 0,
            "red_escalation": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrackGuard-LTE MQTT edge node")
    parser.add_argument("--source", required=True, help="Video file, folder stream URL, or camera index")
    parser.add_argument("--model", default="best_visdrone_full.pt", help="YOLO model path")
    parser.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "hungarian"])
    parser.add_argument("--conf", type=int, default=70, help="Red escalation threshold")
    parser.add_argument("--camera-angle", default="unknown", choices=["oblique", "overhead", "unknown"])
    parser.add_argument("--jetson-id", default=os.getenv("JETSON_ID", "A"))
    parser.add_argument("--camera-id", default=None, help="Camera/node label in MQTT payload")
    parser.add_argument("--mqtt-host", default=os.getenv("MQTT_BROKER_HOST", JetsonConfig.MQTT_BROKER_HOST))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_BROKER_PORT", JetsonConfig.MQTT_BROKER_PORT)))
    parser.add_argument("--device-mode", default="cuda", choices=["cuda", "cpu"], help="Reported device mode")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps-limit", type=float, default=None)
    parser.add_argument("--live-refresh-interval", type=float, default=60.0,
                        help="Proactively reopen live URL/camera sources every N seconds; set 0 to disable")
    parser.add_argument("--heartbeat-interval", type=float, default=5.0, help="Seconds between status MQTT messages")
    parser.add_argument("--evidence-dir", default=None, help="Alarm evidence snapshot directory passed to main.py")
    parser.add_argument("--no-evidence-snapshots", action="store_true")
    parser.add_argument("--stage-profile-dir", default=None, help="Directory for per-stage latency and resource CSVs written by each edge node")
    parser.add_argument("--resource-sample-interval", type=float, default=2.0, help="Seconds between raw edge-resource samples")
    parser.add_argument("--reporting-mode", choices=["mqtt", "local-only"], default="mqtt", help="mqtt publishes event/status metadata; local-only runs the same edge pipeline without MQTT")
    parser.add_argument("--episode-gap-seconds", type=float, default=10.0, help="Maximum time gap in seconds used to merge operator-facing alarms of the same level into one episode")
    parser.add_argument("--episode-gap-frames", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mqtt-smoke-test", action="store_true", help="Publish one status message and exit without running TrackGuard")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.camera_id = args.camera_id or f"camera_{args.jetson_id}"
    if args.episode_gap_frames is not None:
        print("WARNING: --episode-gap-frames is deprecated and ignored; using --episode-gap-seconds instead.")
    profile_dir = Path(args.stage_profile_dir) if args.stage_profile_dir else None
    publish_profile_file = None
    publish_profile_writer = None
    resource_profile_file = None
    resource_profile_writer = None
    args.stage_profile_output = None
    event_profile_file = None
    event_profile_writer = None
    episode_profile_path = None
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        source_id = safe_stem(args.source)
        args.stage_profile_output = str(profile_dir / f"stage_latency_{args.jetson_id}_{source_id}.csv")
        publish_profile_path = profile_dir / f"mqtt_publish_profile_{args.jetson_id}_{source_id}.csv"
        publish_profile_file = open(publish_profile_path, "w", newline="", encoding="utf-8")
        publish_profile_writer = csv.DictWriter(publish_profile_file, fieldnames=[
            "timestamp_epoch", "message_type", "event_index", "frame_id",
            "json_serialize_ms", "mqtt_publish_call_ms", "payload_bytes",
            "mqtt_return_code", "success"
        ])
        publish_profile_writer.writeheader()
        resource_profile_path = profile_dir / f"resource_profile_{args.jetson_id}_{source_id}.csv"
        resource_profile_file = open(resource_profile_path, "w", newline="", encoding="utf-8")
        resource_profile_writer = csv.DictWriter(resource_profile_file, fieldnames=[
            "timestamp_epoch", "runtime_s", "jetson_id", "camera_id",
            "cpu_percent", "gpu_percent", "ram_mb", "power_w",
            "temperature_c"
        ])
        resource_profile_writer.writeheader()
        event_profile_path = profile_dir / f"operator_events_{args.jetson_id}_{source_id}.csv"
        event_profile_file = open(event_profile_path, "w", newline="", encoding="utf-8")
        event_profile_writer = csv.DictWriter(event_profile_file, fieldnames=[
            "timestamp_epoch", "runtime_s", "jetson_id", "camera_id", "frame_id",
            "alarm_level", "severity_index", "event_index", "reporting_mode"
        ])
        event_profile_writer.writeheader()
        episode_profile_path = profile_dir / f"operator_episodes_{args.jetson_id}_{source_id}.csv"

    def record_publish_profile(message: Dict[str, Any], success: bool):
        if publish_profile_writer is None:
            return
        if publisher is None:
            return
        metrics = publisher.get_last_publish_metrics() if hasattr(publisher, "get_last_publish_metrics") else {}
        publish_profile_writer.writerow({
            "timestamp_epoch": f"{time.time():.6f}",
            "message_type": message.get("message_type", ""),
            "event_index": message.get("event_index", ""),
            "frame_id": message.get("frame_id", ""),
            "json_serialize_ms": f"{float(metrics.get('json_serialize_ms', 0.0)):.6f}",
            "mqtt_publish_call_ms": f"{float(metrics.get('mqtt_publish_call_ms', 0.0)):.6f}",
            "payload_bytes": metrics.get("payload_bytes", 0),
            "mqtt_return_code": metrics.get("mqtt_return_code", ""),
            "success": int(bool(success)),
        })
        publish_profile_file.flush()

    folder_videos = list_video_files(args.source)
    if folder_videos:
        return run_folder_mode(args, folder_videos)
    if Path(args.source).is_dir() and not folder_videos:
        print(f"ERROR: no supported video files found in folder: {args.source}")
        print("Supported extensions:", ", ".join(sorted(VIDEO_EXTENSIONS)))
        return 1

    JetsonConfig.JETSON_ID = args.jetson_id
    JetsonConfig.MQTT_BROKER_HOST = args.mqtt_host
    JetsonConfig.MQTT_BROKER_PORT = args.mqtt_port
    publisher_module.JetsonConfig.JETSON_ID = args.jetson_id
    publisher_module.JetsonConfig.MQTT_BROKER_HOST = args.mqtt_host
    publisher_module.JetsonConfig.MQTT_BROKER_PORT = args.mqtt_port

    print("=" * 72)
    print("TrackGuard-LTE Multi-Node Edge Publisher")
    print("=" * 72)
    print(f"Node/Jetson : {args.jetson_id}")
    print(f"Camera ID   : {args.camera_id}")
    print(f"Source      : {args.source}")
    print(f"MQTT broker : {args.mqtt_host}:{args.mqtt_port}")
    print(f"Model       : {args.model}")
    print("=" * 72)

    publisher = None
    if args.reporting_mode == "mqtt":
        publisher = MQTTPublisher(jetson_id=args.jetson_id)
        publisher.broker_host = args.mqtt_host
        publisher.broker_port = args.mqtt_port
        if not publisher.connect():
            print("ERROR: failed to connect to MQTT broker")
            return 1
        print(f"OK: MQTT publisher connected to {args.mqtt_host}:{args.mqtt_port} topic={publisher.topic}")
        if args.mqtt_smoke_test:
            smoke_msg = make_status_message(
                args,
                {"cpu_percent": 0.0, "gpu_percent": 0.0, "ram_mb": 0.0, "power_w": 0.0},
                time.time(),
                frames_processed=1,
                avg_fps=0.0,
                avg_ms_frame=0.0,
            )
            ok = publisher.publish(smoke_msg)
            print("MQTT smoke test waiting 10 seconds for central clock-sync ping...")
            time.sleep(10.0)
            publisher.disconnect()
            print(f"MQTT smoke test publish: {'OK' if ok else 'FAILED'}")
            return 0 if ok else 1
    else:
        print("Reporting    : local-only (MQTT disabled)")

    resource_monitor = ResourceMonitor()
    resource_snapshot: Dict[str, float] = resource_monitor.measure_all()
    resource_lock = threading.Lock()
    stop_monitor = threading.Event()

    def monitor_loop():
        nonlocal resource_snapshot
        while not stop_monitor.is_set():
            try:
                snapshot = resource_monitor.measure_all()
                with resource_lock:
                    resource_snapshot = snapshot
                if resource_profile_writer is not None:
                    resource_profile_writer.writerow({
                        "timestamp_epoch": f"{float(snapshot.get('timestamp', time.time())):.6f}",
                        "runtime_s": f"{max(time.time() - start_time, 0.0):.3f}",
                        "jetson_id": args.jetson_id,
                        "camera_id": args.camera_id,
                        "cpu_percent": f"{float(snapshot.get('cpu_percent', 0.0)):.3f}",
                        "gpu_percent": f"{float(snapshot.get('gpu_percent', 0.0)):.3f}",
                        "ram_mb": f"{float(snapshot.get('ram_mb', 0.0)):.3f}",
                        "power_w": f"{float(snapshot.get('power_w', 0.0)):.3f}",
                        "temperature_c": f"{float(snapshot.get('temperature_c', 0.0)):.3f}",
                    })
                    resource_profile_file.flush()
            except Exception as exc:
                print(f"WARNING: resource profiling sample failed: {exc}")
            time.sleep(max(args.resource_sample_interval, 0.2))

    start_time = time.time()
    thread = threading.Thread(target=monitor_loop, daemon=True)
    thread.start()

    cmd = build_trackguard_command(args)
    print("Running:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))

    event_index = 0
    red_events = 0
    warning_events = 0
    caution_events = 0
    last_heartbeat = 0.0
    frames_processed = None
    avg_fps = None
    avg_ms_frame = None
    if publisher is not None:
        with resource_lock:
            snapshot = dict(resource_snapshot)
        _startup_msg = make_status_message(
            args,
            snapshot,
            start_time,
            frames_processed=0,
            avg_fps=0.0,
            avg_ms_frame=0.0,
        )
        _startup_msg["summary"]["startup"] = True
        _publish_ok = publisher.publish(_startup_msg)
        record_publish_profile(_startup_msg, _publish_ok)
        last_heartbeat = time.time()
        print(f"Startup MQTT status publish: {'OK' if _publish_ok else 'FAILED'}")
    # Online episode merger for operator workload. Episodes are kept separately
    # for critical (warning) and red (emergency) alarm levels and merged by
    # elapsed wall-clock time, because MQTT/operator events are not emitted on
    # every processed frame.
    episode_state = {}
    completed_episodes = []

    def update_episode(level: str, frame_id: int, timestamp: float):
        state = episode_state.get(level)
        gap_seconds = max(float(args.episode_gap_seconds), 0.0)
        if state is None or timestamp - state["end_time"] > gap_seconds:
            if state is not None:
                completed_episodes.append(state)
            episode_state[level] = {
                "alarm_level": level, "start_frame": frame_id, "end_frame": frame_id,
                "start_time": timestamp, "end_time": timestamp, "event_count": 1
            }
        else:
            state["end_frame"] = frame_id
            state["end_time"] = timestamp
            state["event_count"] += 1

    def write_episode_summary():
        if episode_profile_path is None:
            return
        all_eps = list(completed_episodes) + list(episode_state.values())
        with open(episode_profile_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "jetson_id", "camera_id", "alarm_level", "episode_index",
                "start_frame", "end_frame", "start_time_epoch", "end_time_epoch",
                "duration_s", "event_count", "runtime_hours", "episodes_per_hour"
            ])
            writer.writeheader()
            runtime_h = max((time.time() - start_time) / 3600.0, 1e-9)
            totals = {}
            for ep in all_eps:
                totals[ep["alarm_level"]] = totals.get(ep["alarm_level"], 0) + 1
            counters = {}
            for ep in sorted(all_eps, key=lambda x: x["start_time"]):
                level = ep["alarm_level"]
                counters[level] = counters.get(level, 0) + 1
                writer.writerow({
                    "jetson_id": args.jetson_id, "camera_id": args.camera_id,
                    "alarm_level": level, "episode_index": counters[level],
                    "start_frame": ep["start_frame"], "end_frame": ep["end_frame"],
                    "start_time_epoch": f'{ep["start_time"]:.6f}',
                    "end_time_epoch": f'{ep["end_time"]:.6f}',
                    "duration_s": f'{max(ep["end_time"]-ep["start_time"],0.0):.6f}',
                    "event_count": ep["event_count"], "runtime_hours": f'{runtime_h:.6f}',
                    "episodes_per_hour": f'{totals[level]/runtime_h:.6f}',
                })

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'bytetrack'}{os.pathsep}{REPO_ROOT}{os.pathsep}" + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            now = time.time()

            m = ALL_DET_RE.search(line)
            if m:
                event_index += 1
                level = m.group("level").lower()
                if level == "emergency":
                    red_events += 1
                elif level == "warning":
                    warning_events += 1
                elif level == "caution":
                    caution_events += 1
                with resource_lock:
                    snapshot = dict(resource_snapshot)
                message = make_event_message(args, m, snapshot, event_index, start_time)
                update_episode(level, int(m.group("frame")), now)
                if event_profile_writer is not None:
                    event_profile_writer.writerow({
                        "timestamp_epoch": f"{now:.6f}", "runtime_s": f"{now-start_time:.6f}",
                        "jetson_id": args.jetson_id, "camera_id": args.camera_id,
                        "frame_id": int(m.group("frame")), "alarm_level": level,
                        "severity_index": parse_float(m.group("conf")), "event_index": event_index,
                        "reporting_mode": args.reporting_mode,
                    })
                    event_profile_file.flush()
                if publisher is not None:
                    _publish_ok = publisher.publish(message)
                    record_publish_profile(message, _publish_ok)
                continue

            fm = FRAMES_RE.search(line)
            if fm:
                frames_processed = int(fm.group("frames"))
            fps_m = FPS_RE.search(line)
            if fps_m:
                avg_fps = parse_float(fps_m.group("fps"))
            ms_m = MS_RE.search(line)
            if ms_m:
                avg_ms_frame = parse_float(ms_m.group("ms"))

            if publisher is not None and now - last_heartbeat >= args.heartbeat_interval:
                with resource_lock:
                    snapshot = dict(resource_snapshot)
                _status_msg = make_status_message(args, snapshot, start_time)
                _publish_ok = publisher.publish(_status_msg)
                record_publish_profile(_status_msg, _publish_ok)
                last_heartbeat = now

        return_code = proc.wait()
    except KeyboardInterrupt:
        print("\nInterrupted by user, terminating TrackGuard subprocess...")
        proc.terminate()
        return_code = proc.wait(timeout=10)
    finally:
        with resource_lock:
            snapshot = dict(resource_snapshot)
        if publisher is not None:
            _final_msg = make_status_message(
                args, snapshot, start_time, frames_processed=frames_processed,
                avg_fps=avg_fps, avg_ms_frame=avg_ms_frame,
            )
            _publish_ok = publisher.publish(_final_msg)
            record_publish_profile(_final_msg, _publish_ok)
        if publish_profile_file is not None:
            publish_profile_file.close()
        if event_profile_file is not None:
            event_profile_file.close()
        write_episode_summary()
        stop_monitor.set()
        thread.join(timeout=max(args.resource_sample_interval * 2.0, 1.0))
        if resource_profile_file is not None:
            resource_profile_file.close()
        if publisher is not None:
            publisher.disconnect()

    stats = publisher.get_stats() if publisher is not None else {"sent": 0, "failed": 0}
    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print("TrackGuard node summary")
    print("=" * 72)
    print(f"Runtime         : {elapsed:.2f} s")
    print(f"Events parsed   : {event_index}")
    print(f"Red/Warning/Caution: {red_events}/{warning_events}/{caution_events}")
    print(f"MQTT published  : {stats['published']}")
    print(f"MQTT failed     : {stats['failed']}")
    print(f"MQTT bytes      : {stats['total_bytes']}")
    print(f"TrackGuard exit : {return_code}")
    print("=" * 72)
    return int(return_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
