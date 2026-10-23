"""
Timestamp utilities for multi-node evaluation.
"""

import time
from typing import Optional


class TimestampUtils:
    """Utilities for timestamp management."""
    
    @staticmethod
    def get_current_timestamp() -> float:
        """Get current Unix timestamp."""
        return time.time()
    
    @staticmethod
    def calculate_latency(send_time: float, receive_time: float) -> float:
        """Calculate latency in milliseconds."""
        return (receive_time - send_time) * 1000.0
    
    @staticmethod
    def calculate_end_to_end_latency(capture_time: float, receive_time: float) -> float:
        """Calculate end-to-end latency from capture to receive."""
        return (receive_time - capture_time) * 1000.0
    
    @staticmethod
    def format_timestamp(timestamp: float) -> str:
        """Format timestamp for logging."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
