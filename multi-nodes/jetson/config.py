"""
Configuration for Jetson devices.
"""

import os
from typing import Optional

class JetsonConfig:
    """Configuration for Jetson device."""
    
    # Jetson ID (A, B, or C)
    JETSON_ID: str = os.getenv("JETSON_ID", "A")
    
    # MQTT Broker Configuration
    MQTT_BROKER_HOST: str = os.getenv("MQTT_BROKER_HOST", "localhost")
    MQTT_BROKER_PORT: int = int(os.getenv("MQTT_BROKER_PORT", "1883"))
    MQTT_TOPIC_PREFIX: str = f"jetson_{JETSON_ID}"
    MQTT_TOPIC_RESULTS: str = f"{MQTT_TOPIC_PREFIX}/results"
    MQTT_QOS: int = 1  # At least once delivery
    
    # Resource Monitoring
    RESOURCE_MONITOR_INTERVAL: float = 1.0  # seconds
    
    # Video Configuration
    DEFAULT_VIDEO_PATH: str = "video/barbecue.mp4"
    
    @classmethod
    def get_mqtt_client_id(cls) -> str:
        """Get unique MQTT client ID."""
        return f"trackguard_jetson_{cls.JETSON_ID}"
