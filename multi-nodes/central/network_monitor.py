"""
Network performance monitoring for multi-node evaluation.
Thread-safe for concurrent access from multiple Jetsons.
"""

import time
import threading
from typing import Dict, List, Optional
from collections import defaultdict, deque

# Import timestamp utils using direct file import
import importlib.util
from pathlib import Path

timestamp_path = Path(__file__).parent.parent / "utils" / "timestamp_utils.py"
spec = importlib.util.spec_from_file_location("timestamp_utils", timestamp_path)
timestamp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timestamp_module)
TimestampUtils = timestamp_module.TimestampUtils


class NetworkMonitor:
    """Monitor network performance metrics (thread-safe)."""
    
    def __init__(self, window_size: int = 100, clock_offsets_ms: Optional[Dict[str, float]] = None):
        self.window_size = window_size
        self.latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.end_to_end_latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.bandwidth_samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.message_sizes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.message_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        self.total_bytes: Dict[str, int] = defaultdict(int)
        self.start_time: Optional[float] = None
        self.clock_offsets_ms = clock_offsets_ms or {}
        self._lock = threading.Lock()  # Thread-safe access
    
    def update_clock_offset(self, jetson_id: str, offset_ms: float):
        """Update measured Jetson-minus-server clock offset in milliseconds."""
        with self._lock:
            self.clock_offsets_ms[jetson_id] = float(offset_ms)

    def get_clock_offsets(self) -> Dict[str, float]:
        """Return current measured clock offsets."""
        with self._lock:
            return dict(self.clock_offsets_ms)

    def record_message(self, message: Dict[str, any], message_size_bytes: int):
        """Record a received message for network analysis (thread-safe)."""
        with self._lock:
            jetson_id = message.get("jetson_id", "unknown")
            timestamps = message.get("timestamps", {})
            receive_time = message.get("receive_time", time.time())
            clock_offset_ms = float(self.clock_offsets_ms.get(jetson_id, self.clock_offsets_ms.get("all", 0.0)))
            receive_time_corrected = receive_time + (clock_offset_ms / 1000.0)
            
            if self.start_time is None:
                self.start_time = receive_time
            
            # Calculate latencies
            if "send_time" in timestamps and "capture_time" in timestamps:
                send_time = timestamps["send_time"]
                capture_time = timestamps["capture_time"]
                
                # Network latency (send to receive)
                network_latency = TimestampUtils.calculate_latency(send_time, receive_time_corrected)
                self.latencies[jetson_id].append(network_latency)
                
                # End-to-end latency (capture to receive)
                e2e_latency = TimestampUtils.calculate_end_to_end_latency(capture_time, receive_time_corrected)
                self.end_to_end_latencies[jetson_id].append(e2e_latency)
            
            # Record bandwidth data
            self.message_sizes[jetson_id].append(message_size_bytes)
            self.message_times[jetson_id].append(receive_time)
            self.total_bytes[jetson_id] += message_size_bytes
    
    def get_network_latency_stats(self, jetson_id: str) -> Dict[str, float]:
        """Get network latency statistics for a Jetson (thread-safe)."""
        with self._lock:
            if jetson_id not in self.latencies or len(self.latencies[jetson_id]) == 0:
                return {
                    "avg_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "count": 0
                }
            
            latencies = list(self.latencies[jetson_id])  # Copy to avoid holding lock during computation
            return {
                "avg_ms": sum(latencies) / len(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "count": len(latencies)
            }
    
    def get_end_to_end_latency_stats(self, jetson_id: str) -> Dict[str, float]:
        """Get end-to-end latency statistics for a Jetson (thread-safe)."""
        with self._lock:
            if jetson_id not in self.end_to_end_latencies or len(self.end_to_end_latencies[jetson_id]) == 0:
                return {
                    "avg_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "count": 0
                }
            
            latencies = list(self.end_to_end_latencies[jetson_id])  # Copy to avoid holding lock during computation
            return {
                "avg_ms": sum(latencies) / len(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "count": len(latencies)
            }
    
    def get_bandwidth_stats(self, jetson_id: str) -> Dict[str, float]:
        """Get bandwidth statistics for a Jetson (thread-safe)."""
        with self._lock:
            if jetson_id not in self.message_times or len(self.message_times[jetson_id]) < 2:
                return {
                    "bytes_per_sec": 0.0,
                    "kbps": 0.0,
                    "total_bytes": self.total_bytes.get(jetson_id, 0)
                }
            
            times = list(self.message_times[jetson_id])  # Copy to avoid holding lock during computation
            sizes = list(self.message_sizes[jetson_id])
            total_bytes_val = self.total_bytes.get(jetson_id, 0)
            
            if len(times) < 2:
                return {
                    "bytes_per_sec": 0.0,
                    "kbps": 0.0,
                    "total_bytes": total_bytes_val
                }
            
            time_span = times[-1] - times[0]
            total_bytes = sum(sizes)
            
            if time_span > 0:
                bytes_per_sec = total_bytes / time_span
                kbps = (bytes_per_sec * 8) / 1000.0  # Convert to kbps
            else:
                bytes_per_sec = 0.0
                kbps = 0.0
            
            return {
                "bytes_per_sec": bytes_per_sec,
                "kbps": kbps,
                "total_bytes": total_bytes_val
            }
    
    def get_message_throughput(self, jetson_id: str) -> float:
        """Get message throughput (messages per second) for a Jetson (thread-safe)."""
        with self._lock:
            if jetson_id not in self.message_times or len(self.message_times[jetson_id]) < 2:
                return 0.0
            
            times = list(self.message_times[jetson_id])  # Copy to avoid holding lock during computation
            time_span = times[-1] - times[0]
            
            if time_span > 0:
                return len(times) / time_span
            return 0.0
    
    def get_all_stats(self) -> Dict[str, Dict[str, any]]:
        """Get statistics for all Jetsons."""
        all_jetsons = set(list(self.latencies.keys()) + 
                         list(self.end_to_end_latencies.keys()) + 
                         list(self.message_times.keys()))
        
        stats = {}
        for jetson_id in all_jetsons:
            stats[jetson_id] = {
                "network_latency": self.get_network_latency_stats(jetson_id),
                "end_to_end_latency": self.get_end_to_end_latency_stats(jetson_id),
                "bandwidth": self.get_bandwidth_stats(jetson_id),
                "throughput_msg_per_sec": self.get_message_throughput(jetson_id)
            }
        
        return stats

