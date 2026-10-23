"""
MQTT Subscriber for central PC.
Subscribes to results from all Jetson devices.
"""

import json
import logging
import os
import queue
import time
from typing import Dict, Any, Callable, Optional
import paho.mqtt.client as mqtt


class MQTTSubscriber:
    """MQTT subscriber for receiving results from Jetson devices."""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = None
        self.client_id = f"trackguard_central_pc_{os.getpid()}_{int(time.time() * 1000)}"
        self.connected = False
        self.message_queue = queue.Queue()
        self.message_count = 0
        self.callback: Optional[Callable] = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker."""
        if rc == 0:
            self.connected = True
            self.logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            subscriptions = [
                ("+/results", 1),
                ("trackguard/+/results", 1),
                ("trackguard/results/+", 1),
                ("trackguard/sync_response", 1),
            ]
            client.subscribe(subscriptions)
            self.logger.info(
                "Subscribed to topics: +/results, trackguard/+/results, "
                "trackguard/results/+, trackguard/sync_response"
            )
        else:
            self.connected = False
            self.logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker."""
        self.connected = False
        if rc != 0:
            self.logger.warning(f"Unexpected disconnection from MQTT broker. Return code: {rc}")
        else:
            self.logger.info("Disconnected from MQTT broker")
    
    def _on_message(self, client, userdata, msg):
        """Callback when message is received."""
        try:
            receive_time = time.time()
            message = json.loads(msg.payload.decode())
            message["receive_time"] = receive_time
            message["_mqtt_topic"] = msg.topic

            if msg.topic.startswith("trackguard/sync/") and message.get("message_type") != "sync_response":
                return
            
            self.message_count += 1
            
            # Call callback if set
            if self.callback:
                self.callback(message)
            else:
                # Backup path for legacy callers that poll get_message().
                self.message_queue.put(message)
            
            if self.message_count == 1 or self.message_count % 100 == 0:
                self.logger.info(
                    "Received message #%d topic=%s type=%s jetson=%s frame=%s",
                    self.message_count,
                    msg.topic,
                    message.get("message_type", ""),
                    message.get("jetson_id", ""),
                    message.get("frame_id", ""),
                )
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    def set_callback(self, callback: Callable):
        """Set callback function for received messages."""
        self.callback = callback
    
    def connect(self) -> bool:
        """Connect to MQTT broker."""
        try:
            self.client = mqtt.Client(client_id=self.client_id)
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            
            self.logger.info(f"Connecting to MQTT broker {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 5
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            return self.connected
        except Exception as e:
            self.logger.error(f"Error connecting to MQTT broker: {e}")
            return False
    
    def publish_json(self, topic: str, message: Dict[str, Any], qos: int = 1) -> bool:
        """Publish a JSON message from the central client."""
        if not self.client or not self.connected:
            return False
        payload = json.dumps(message, default=str, separators=(",", ":"))
        result = self.client.publish(topic, payload, qos=qos)
        return result.rc == mqtt.MQTT_ERR_SUCCESS

    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            self.logger.info("Disconnected from MQTT broker")
    
    def get_message(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get message from queue."""
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get subscription statistics."""
        return {
            "received": self.message_count,
            "queue_size": self.message_queue.qsize()
        }
