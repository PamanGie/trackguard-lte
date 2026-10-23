"""
Results logger for multi-node evaluation.
Exports results to CSV files.
Thread-safe for concurrent access from multiple Jetsons.
"""

import csv
import os
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class ResultsLogger:
    """Log results to CSV files (thread-safe)."""
    
    def __init__(self, output_dir: str = "multi_node_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV file paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"multi_node_results_{timestamp}.csv"
        self.network_file = self.output_dir / f"network_performance_{timestamp}.csv"
        self.resource_file = self.output_dir / f"resource_usage_{timestamp}.csv"
        self.alarm_file = self.output_dir / f"alarm_events_{timestamp}.csv"
        
        # CSV writers
        self.results_writer = None
        self.network_writer = None
        self.resource_writer = None
        self.alarm_writer = None
        
        self.results_file_handle = None
        self.network_file_handle = None
        self.resource_file_handle = None
        self.alarm_file_handle = None
        
        # Thread-safe access
        self._lock = threading.Lock()

    @staticmethod
    def _iso(epoch_value: Any) -> str:
        try:
            value = float(epoch_value)
            if value <= 0:
                return ""
            return datetime.fromtimestamp(value).isoformat(timespec="seconds")
        except Exception:
            return ""
    
    def _init_results_csv(self):
        """Initialize results CSV file."""
        self.results_file_handle = open(self.results_file, 'w', newline='')
        self.results_writer = csv.DictWriter(
            self.results_file_handle,
            fieldnames=[
                "timestamp", "jetson_id", "frame_id",
                "message_type", "camera_id", "alarm_level", "severity_index",
                "label", "mode", "num_detections", "num_confirmed", "num_monitoring",
                "capture_time", "send_time", "receive_time",
                "network_latency_ms", "end_to_end_latency_ms",
                "payload_bytes", "source"
            ]
        )
        self.results_writer.writeheader()

    def _init_alarm_csv(self):
        """Initialize alarm event CSV file."""
        self.alarm_file_handle = open(self.alarm_file, 'w', newline='')
        self.alarm_writer = csv.DictWriter(
            self.alarm_file_handle,
            fieldnames=[
                "timestamp", "timestamp_iso", "jetson_id", "camera_id", "source", "frame_id",
                "alarm_level", "severity_index", "label", "mode",
                "kelr_i", "kelr_j", "ars_zi", "ars_zj", "iou",
                "network_latency_ms", "end_to_end_latency_ms", "payload_bytes"
            ]
        )
        self.alarm_writer.writeheader()
    
    def _init_network_csv(self):
        """Initialize network performance CSV file."""
        self.network_file_handle = open(self.network_file, 'w', newline='')
        self.network_writer = csv.DictWriter(
            self.network_file_handle,
            fieldnames=[
                "jetson_id", "avg_network_latency_ms", "min_network_latency_ms",
                "max_network_latency_ms", "avg_e2e_latency_ms", "min_e2e_latency_ms",
                "max_e2e_latency_ms", "bandwidth_kbps", "throughput_msg_per_sec",
                "total_bytes", "sample_count"
            ]
        )
        self.network_writer.writeheader()
    
    def _init_resource_csv(self):
        """Initialize resource usage CSV file."""
        self.resource_file_handle = open(self.resource_file, 'w', newline='')
        self.resource_writer = csv.DictWriter(
            self.resource_file_handle,
            fieldnames=[
                "jetson_id", "avg_gpu_percent", "max_gpu_percent",
                "avg_ram_mb", "max_ram_mb", "avg_power_w", "max_power_w",
                "avg_cpu_percent", "total_frames", "avg_detections_per_frame",
                "avg_confirmed_per_frame", "avg_monitoring_per_frame",
                "total_alarm_events", "red_events", "warning_events", "caution_events"
            ]
        )
        self.resource_writer.writeheader()
    
    def log_frame_result(self, message: Dict[str, Any], network_stats: Dict[str, float]):
        """Log a single frame result (thread-safe)."""
        with self._lock:
            if self.results_writer is None:
                self._init_results_csv()
            
            timestamps = message.get("timestamps", {})
            receive_time = message.get("receive_time", 0)
            
            network_latency = network_stats.get("network_latency_ms", 0.0)
            e2e_latency = network_stats.get("end_to_end_latency_ms", 0.0)
            payload_bytes = network_stats.get("payload_bytes", 0)
            summary = message.get("summary", {})
            
            self.results_writer.writerow({
            "timestamp": receive_time,
            "jetson_id": message.get("jetson_id", "unknown"),
            "frame_id": message.get("frame_id", 0),
            "message_type": message.get("message_type", "legacy"),
            "camera_id": message.get("camera_id", ""),
            "alarm_level": message.get("alarm_level", ""),
            "severity_index": message.get("severity_index", ""),
            "label": message.get("label", ""),
            "mode": message.get("mode", ""),
            "num_detections": summary.get("num_detections", 0),
            "num_confirmed": summary.get("num_confirmed", 0),
            "num_monitoring": summary.get("num_monitoring", 0),
            "capture_time": timestamps.get("capture_time", 0),
            "send_time": timestamps.get("send_time", 0),
            "receive_time": receive_time,
                "network_latency_ms": network_latency,
                "end_to_end_latency_ms": e2e_latency,
                "payload_bytes": payload_bytes,
                "source": message.get("source", "")
            })
            if message.get("message_type") == "alarm":
                self.log_alarm_event(message, network_stats, _lock_already_held=True)
            # Flush with error handling (file might be open in Excel)
            try:
                self.results_file_handle.flush()
            except (PermissionError, IOError) as e:
                # File might be open in Excel - continue without flush
                # Data will be written on close
                pass

    def log_alarm_event(self, message: Dict[str, Any], network_stats: Dict[str, float], _lock_already_held: bool = False):
        """Log one TrackGuard alarm event."""
        def _write():
            if self.alarm_writer is None:
                self._init_alarm_csv()
            timestamps = message.get("timestamps", {})
            receive_time = message.get("receive_time", 0)
            evidence = message.get("evidence", {})
            self.alarm_writer.writerow({
                "timestamp": receive_time,
                "timestamp_iso": self._iso(receive_time),
                "jetson_id": message.get("jetson_id", "unknown"),
                "camera_id": message.get("camera_id", ""),
                "source": message.get("source", ""),
                "frame_id": message.get("frame_id", 0),
                "alarm_level": message.get("alarm_level", ""),
                "severity_index": message.get("severity_index", ""),
                "label": message.get("label", ""),
                "mode": message.get("mode", ""),
                "kelr_i": evidence.get("kelr_i", ""),
                "kelr_j": evidence.get("kelr_j", ""),
                "ars_zi": evidence.get("ars_zi", ""),
                "ars_zj": evidence.get("ars_zj", ""),
                "iou": evidence.get("iou", ""),
                "network_latency_ms": network_stats.get("network_latency_ms", 0.0),
                "end_to_end_latency_ms": network_stats.get("end_to_end_latency_ms", 0.0),
                "payload_bytes": network_stats.get("payload_bytes", 0),
            })
            try:
                self.alarm_file_handle.flush()
            except (PermissionError, IOError):
                pass

        if _lock_already_held:
            _write()
        else:
            with self._lock:
                _write()
    
    def log_network_stats(self, network_stats: Dict[str, Dict[str, Any]]):
        """Log network statistics for all Jetsons."""
        if self.network_writer is None:
            self._init_network_csv()
        
        for jetson_id, stats in network_stats.items():
            net_lat = stats.get("network_latency", {})
            e2e_lat = stats.get("end_to_end_latency", {})
            bandwidth = stats.get("bandwidth", {})
            
            self.network_writer.writerow({
                "jetson_id": jetson_id,
                "avg_network_latency_ms": net_lat.get("avg_ms", 0.0),
                "min_network_latency_ms": net_lat.get("min_ms", 0.0),
                "max_network_latency_ms": net_lat.get("max_ms", 0.0),
                "avg_e2e_latency_ms": e2e_lat.get("avg_ms", 0.0),
                "min_e2e_latency_ms": e2e_lat.get("min_ms", 0.0),
                "max_e2e_latency_ms": e2e_lat.get("max_ms", 0.0),
                "bandwidth_kbps": bandwidth.get("kbps", 0.0),
                "throughput_msg_per_sec": stats.get("throughput_msg_per_sec", 0.0),
                "total_bytes": bandwidth.get("total_bytes", 0),
                "sample_count": net_lat.get("count", 0)
            })
        # Flush with error handling
        try:
            self.network_file_handle.flush()
        except (PermissionError, IOError):
            pass
    
    def log_resource_stats(self, resource_stats: Dict[str, Dict[str, Any]]):
        """Log resource statistics for all Jetsons."""
        if self.resource_writer is None:
            self._init_resource_csv()
        
        for jetson_id, stats in resource_stats.items():
            resource = stats.get("resource", {})
            processing = stats.get("processing", {})
            alarms = stats.get("alarms", {})
            
            self.resource_writer.writerow({
                "jetson_id": jetson_id,
                "avg_gpu_percent": resource.get("avg_gpu_percent", 0.0),
                "max_gpu_percent": resource.get("max_gpu_percent", 0.0),
                "avg_ram_mb": resource.get("avg_ram_mb", 0.0),
                "max_ram_mb": resource.get("max_ram_mb", 0.0),
                "avg_power_w": resource.get("avg_power_w", 0.0),
                "max_power_w": resource.get("max_power_w", 0.0),
                "avg_cpu_percent": resource.get("avg_cpu_percent", 0.0),
                "total_frames": processing.get("total_frames", 0),
                "avg_detections_per_frame": processing.get("avg_detections_per_frame", 0.0),
                "avg_confirmed_per_frame": processing.get("avg_confirmed_per_frame", 0.0),
                "avg_monitoring_per_frame": processing.get("avg_monitoring_per_frame", 0.0),
                "total_alarm_events": alarms.get("total_alarm_events", 0),
                "red_events": alarms.get("red_events", 0),
                "warning_events": alarms.get("warning_events", 0),
                "caution_events": alarms.get("caution_events", 0)
            })
        # Flush with error handling
        try:
            self.resource_file_handle.flush()
        except (PermissionError, IOError):
            pass
    
    def close(self):
        """Close all CSV files with error handling."""
        # Close results file
        if self.results_file_handle:
            try:
                self.results_file_handle.flush()
                self.results_file_handle.close()
            except (PermissionError, IOError) as e:
                print(f"WARNING: Could not close results file: {e}")
                print(f"   File may be open in Excel or another application.")
                print(f"   File location: {self.results_file}")
        
        # Close network file
        if self.network_file_handle:
            try:
                self.network_file_handle.flush()
                self.network_file_handle.close()
            except (PermissionError, IOError) as e:
                print(f"WARNING: Could not close network file: {e}")
                print(f"   File location: {self.network_file}")
        
        # Close resource file
        if self.resource_file_handle:
            try:
                self.resource_file_handle.flush()
                self.resource_file_handle.close()
            except (PermissionError, IOError) as e:
                print(f"WARNING: Could not close resource file: {e}")
                print(f"   File location: {self.resource_file}")
        if self.alarm_file_handle:
            try:
                self.alarm_file_handle.flush()
                self.alarm_file_handle.close()
            except (PermissionError, IOError) as e:
                print(f"WARNING: Could not close alarm file: {e}")
                print(f"   File location: {self.alarm_file}")
    
    def get_output_files(self) -> Dict[str, str]:
        """Get paths to output files."""
        return {
            "results": str(self.results_file),
            "network": str(self.network_file),
            "resource": str(self.resource_file),
            "alarms": str(self.alarm_file)
        }
