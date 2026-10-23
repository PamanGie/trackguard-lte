"""
Data aggregator for multi-node evaluation.
Aggregates resource metrics from all Jetson devices.
Thread-safe for concurrent access from multiple Jetsons.
"""

import threading
from typing import Dict, List, Any
from collections import defaultdict


class DataAggregator:
    """Aggregate data from multiple Jetson devices (thread-safe)."""
    
    def __init__(self):
        self.resource_samples: Dict[str, List[Dict]] = defaultdict(list)
        self.frame_results: Dict[str, List[Dict]] = defaultdict(list)
        self.alarm_events: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.Lock()  # Thread-safe access
    
    def add_message(self, message: Dict[str, Any]):
        """Add a message from a Jetson device (thread-safe)."""
        jetson_id = message.get("jetson_id", "unknown")
        
        with self._lock:
            node_metrics = message.get("node_metrics", {})
            resource_metrics = node_metrics.get("resource")
            if resource_metrics:
                self.resource_samples[jetson_id].append(resource_metrics)
            if "resource_metrics" in message:
                self.resource_samples[jetson_id].append(message["resource_metrics"])

            if message.get("message_type") == "alarm":
                self.alarm_events[jetson_id].append(message)

            # Store frame results
            self.frame_results[jetson_id].append({
                "frame_id": message.get("frame_id", 0),
                "num_detections": message.get("summary", {}).get("num_detections", 0),
                "num_confirmed": message.get("summary", {}).get("num_confirmed", 0),
                "num_monitoring": message.get("summary", {}).get("num_monitoring", 0),
                "message_type": message.get("message_type", "legacy"),
                "alarm_level": message.get("alarm_level", "")
            })
    
    def get_resource_stats(self, jetson_id: str) -> Dict[str, float]:
        """Get aggregated resource statistics for a Jetson (thread-safe)."""
        with self._lock:
            if jetson_id not in self.resource_samples or len(self.resource_samples[jetson_id]) == 0:
                return {
                    "avg_gpu_percent": 0.0,
                    "avg_ram_mb": 0.0,
                    "avg_power_w": 0.0,
                    "avg_cpu_percent": 0.0,
                    "max_gpu_percent": 0.0,
                    "max_ram_mb": 0.0,
                    "max_power_w": 0.0,
                    "sample_count": 0
                }
            
            samples = self.resource_samples[jetson_id].copy()  # Copy to avoid holding lock during computation
        
        gpu_values = [s.get("gpu_percent", 0.0) for s in samples]
        ram_values = [s.get("ram_mb", 0.0) for s in samples]
        power_values = [s.get("power_w", 0.0) for s in samples]
        cpu_values = [s.get("cpu_percent", 0.0) for s in samples]
        
        return {
            "avg_gpu_percent": sum(gpu_values) / len(gpu_values) if gpu_values else 0.0,
            "avg_ram_mb": sum(ram_values) / len(ram_values) if ram_values else 0.0,
            "avg_power_w": sum(power_values) / len(power_values) if power_values else 0.0,
            "avg_cpu_percent": sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
            "max_gpu_percent": max(gpu_values) if gpu_values else 0.0,
            "max_ram_mb": max(ram_values) if ram_values else 0.0,
            "max_power_w": max(power_values) if power_values else 0.0,
            "sample_count": len(samples)
        }
    
    def get_processing_stats(self, jetson_id: str) -> Dict[str, float]:
        """Get processing statistics for a Jetson (thread-safe)."""
        with self._lock:
            if jetson_id not in self.frame_results or len(self.frame_results[jetson_id]) == 0:
                return {
                    "total_frames": 0,
                    "avg_detections_per_frame": 0.0,
                    "avg_confirmed_per_frame": 0.0,
                    "avg_monitoring_per_frame": 0.0
                }
            
            frames = self.frame_results[jetson_id].copy()  # Copy to avoid holding lock during computation
        
        total_detections = sum(f["num_detections"] for f in frames)
        total_confirmed = sum(f["num_confirmed"] for f in frames)
        total_monitoring = sum(f["num_monitoring"] for f in frames)
        
        return {
            "total_frames": len(frames),
            "avg_detections_per_frame": total_detections / len(frames) if frames else 0.0,
            "avg_confirmed_per_frame": total_confirmed / len(frames) if frames else 0.0,
            "avg_monitoring_per_frame": total_monitoring / len(frames) if frames else 0.0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated statistics for all Jetsons (thread-safe)."""
        with self._lock:
            all_jetsons = set(list(self.resource_samples.keys()) + list(self.frame_results.keys()) + list(self.alarm_events.keys()))
        
        stats = {}
        for jetson_id in all_jetsons:
            stats[jetson_id] = {
                "resource": self.get_resource_stats(jetson_id),
                "processing": self.get_processing_stats(jetson_id),
                "alarms": self.get_alarm_stats(jetson_id)
            }
        
        return stats

    def get_alarm_stats(self, jetson_id: str) -> Dict[str, Any]:
        """Get alarm counts for a Jetson."""
        with self._lock:
            events = list(self.alarm_events.get(jetson_id, []))
        levels = defaultdict(int)
        for event in events:
            levels[event.get("alarm_level", "unknown")] += 1
        return {
            "total_alarm_events": len(events),
            "red_events": levels.get("emergency", 0),
            "warning_events": levels.get("warning", 0),
            "caution_events": levels.get("caution", 0),
        }
    
    def get_total_stats(self) -> Dict[str, float]:
        """Get total aggregated statistics across all Jetsons."""
        all_stats = self.get_all_stats()
        
        total_power = sum(s["resource"]["avg_power_w"] for s in all_stats.values())
        total_ram = sum(s["resource"]["avg_ram_mb"] for s in all_stats.values())
        total_frames = sum(s["processing"]["total_frames"] for s in all_stats.values())
        
        avg_gpu = sum(s["resource"]["avg_gpu_percent"] for s in all_stats.values()) / len(all_stats) if all_stats else 0.0
        
        return {
            "total_power_w": total_power,
            "total_ram_mb": total_ram,
            "avg_gpu_percent": avg_gpu,
            "total_frames": total_frames,
            "num_jetsons": len(all_stats)
        }
