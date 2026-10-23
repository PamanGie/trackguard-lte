"""
MQTT Publisher for Jetson devices.
Publishes LTE-Fire results to MQTT broker.
"""

import json
import time
import logging
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional
import paho.mqtt.client as mqtt

# Import config using direct file import
config_path = Path(__file__).parent / "config.py"
spec = importlib.util.spec_from_file_location("jetson_config", config_path)
jetson_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jetson_config)
JetsonConfig = jetson_config.JetsonConfig


class MQTTPublisher:
    """MQTT publisher for sending results to central PC."""
    
    def __init__(self, jetson_id: Optional[str] = None):
        self.jetson_id = jetson_id or JetsonConfig.JETSON_ID
        self.broker_host = JetsonConfig.MQTT_BROKER_HOST
        self.broker_port = JetsonConfig.MQTT_BROKER_PORT
        self.topic = f"jetson_{self.jetson_id}/results"
        self.qos = JetsonConfig.MQTT_QOS
        
        self.client = None
        self.connected = False
        self.publish_count = 0
        self.failed_count = 0
        self.total_bytes = 0
        self.last_payload_bytes = 0
        self.last_publish_metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker."""
        if rc == 0:
            self.connected = True
            self.logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            client.subscribe(f"trackguard/sync/{self.jetson_id}", qos=1)
            client.subscribe("trackguard/sync/all", qos=1)
            self.logger.info(
                "Subscribed to sync topics: trackguard/sync/%s, trackguard/sync/all",
                self.jetson_id,
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
    
    def _on_publish(self, client, userdata, mid):
        """Callback when message is published."""
        self.publish_count += 1
        if self.publish_count % 100 == 0:
            self.logger.info(f"Published {self.publish_count} messages")

    def _on_message(self, client, userdata, msg):
        """Respond to central clock-sync pings."""
        try:
            payload = json.loads(msg.payload.decode())
            if payload.get("message_type") != "sync_ping":
                return
            t_recv = time.time()
            response = {
                "message_type": "sync_response",
                "jetson_id": self.jetson_id,
                "server_send_time": payload.get("server_send_time"),
                "sync_seq": payload.get("sync_seq"),
                "jetson_receive_time": t_recv,
                "jetson_send_time": time.time(),
            }
            info = self.client.publish(
                "trackguard/sync_response",
                json.dumps(response, separators=(",", ":")),
                qos=1,
            )
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                self.logger.warning("Failed to queue sync response. Return code: %s", info.rc)
        except Exception as e:
            self.logger.warning(f"Error handling sync message: {e}")
    
    def _cleanup_client(self):
        """Stop and disconnect existing client before reconnect."""
        if self.client is None:
            return
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
        self.client = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to MQTT broker."""
        try:
            self._cleanup_client()
            # Unique client_id per connection to avoid broker kicking old session (rc=7)
            base_id = JetsonConfig.get_mqtt_client_id()
            client_id = f"{base_id}_{int(time.time() * 1000)}"
            self.client = mqtt.Client(client_id=client_id)
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_publish = self._on_publish
            self.client.on_message = self._on_message

            self.logger.info(f"Connecting to MQTT broker {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()

            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            return self.connected
        except Exception as e:
            self.logger.error(f"Error connecting to MQTT broker: {e}")
            self._cleanup_client()
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
            self.client = None
            self.connected = False
            self.logger.info("Disconnected from MQTT broker")
    
    def publish(self, message: Dict[str, Any]) -> bool:
        """Publish message to MQTT broker."""
        if not self.connected:
            self.logger.warning("Not connected to MQTT broker. Attempting to reconnect...")
            if not self.connect():
                self.failed_count += 1
                return False
        
        try:
            # Convert message to JSON
            serialize_start = time.perf_counter()
            json_message = json.dumps(message, default=str, separators=(",", ":"))
            payload_bytes = len(json_message.encode("utf-8"))
            serialize_ms = (time.perf_counter() - serialize_start) * 1000.0
            
            # Publish
            publish_start = time.perf_counter()
            result = self.client.publish(
                self.topic,
                json_message,
                qos=self.qos
            )
            published = False
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                result.wait_for_publish(timeout=5.0)
                published = result.is_published()
            publish_call_ms = (time.perf_counter() - publish_start) * 1000.0
            self.last_publish_metrics = {
                "json_serialize_ms": serialize_ms,
                "mqtt_publish_call_ms": publish_call_ms,
                "payload_bytes": payload_bytes,
                "mqtt_return_code": int(result.rc),
                "mqtt_published": int(published),
            }
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.last_payload_bytes = payload_bytes
                self.total_bytes += payload_bytes
                return True
            else:
                self.failed_count += 1
                self.logger.error(
                    f"Failed to publish message. Return code: {result.rc}, "
                    f"published={result.is_published()}"
                )
                return False
        except Exception as e:
            self.failed_count += 1
            self.logger.error(f"Error publishing message: {e}")
            return False
    
    def get_last_publish_metrics(self) -> Dict[str, Any]:
        """Get latency metrics for the most recent publish call."""
        return dict(self.last_publish_metrics)

    def get_stats(self) -> Dict[str, int]:
        """Get publishing statistics."""
        return {
            "published": self.publish_count,
            "failed": self.failed_count,
            "total": self.publish_count + self.failed_count,
            "total_bytes": self.total_bytes,
            "last_payload_bytes": self.last_payload_bytes
        }
