"""
Message formatter for MQTT communication.
Converts LTE-Fire results to JSON format.
"""

import json
import time
from typing import Dict, List, Any, Optional
from lte_fire.lte_module import FireParticle


class MessageFormatter:
    """Format LTE-Fire results for MQTT transmission."""
    
    @staticmethod
    def format_detection(detection: Dict) -> Dict:
        """Format a single detection."""
        return {
            "bbox": detection.get("bbox", []),
            "confidence": detection.get("confidence", 0.0),
            "class_id": detection.get("class_id", 0)
        }
    
    @staticmethod
    def format_particle(particle: FireParticle) -> Dict:
        """Format a single fire particle."""
        return {
            "particle_id": particle.particle_id,
            "bbox": list(particle.bbox),
            "center": list(particle.center),
            "velocity": list(particle.velocity) if hasattr(particle, 'velocity') else [0.0, 0.0],
            "confidence": particle.confidence,
            "state": particle.state.name if hasattr(particle.state, 'name') else str(particle.state),
            "size": particle.mass if hasattr(particle, 'mass') else 0.0,
            "age": particle.age if hasattr(particle, 'age') else 0
        }
    
    @staticmethod
    def format_physics_metrics(particle: FireParticle) -> Dict:
        """Extract physics metrics from particle."""
        metrics = {
            "divergence": 0.0,
            "shear": 0.0,
            "flow_continuity": 0.0,
            "size_change_rate": 0.0,
            "spread_direction": None
        }
        
        # Try to extract physics metrics if available
        if hasattr(particle, 'physics_scores'):
            metrics.update(particle.physics_scores)
        
        return metrics
    
    @staticmethod
    def format_behavior_analytics(particle: FireParticle) -> Dict:
        """Extract behavior analytics from particle (size change, spreading direction)."""
        import numpy as np
        
        behavior = {
            "size_status": "STABLE",
            "size_change_pct": 0.0,
            "spread_direction": "STATIC",
            "spread_angle": 0.0,
            "spread_speed": 0.0
        }
        
        # Extract size status (GROWING/SHRINKING/STABLE)
        if hasattr(particle, 'size_status'):
            behavior["size_status"] = particle.size_status
        if hasattr(particle, 'size_change_pct'):
            behavior["size_change_pct"] = float(particle.size_change_pct)
        
        # Extract spreading direction from velocity
        vx, vy = particle.velocity if hasattr(particle, 'velocity') else (0.0, 0.0)
        speed = np.sqrt(vx**2 + vy**2)
        behavior["spread_speed"] = float(speed)
        
        # Movement threshold (pixels per frame)
        MOVEMENT_THRESHOLD = 2.0
        
        if speed >= MOVEMENT_THRESHOLD:
            # Calculate angle in degrees (0 = right, 90 = down, 180 = left, 270 = up)
            angle_rad = np.arctan2(vy, vx)
            angle_deg = np.degrees(angle_rad)
            behavior["spread_angle"] = float(angle_deg)
            
            # Determine cardinal direction
            if abs(vx) > abs(vy):
                behavior["spread_direction"] = "RIGHT" if vx > 0 else "LEFT"
            else:
                behavior["spread_direction"] = "DOWN" if vy > 0 else "UP"
        else:
            behavior["spread_direction"] = "STATIC"
            behavior["spread_angle"] = 0.0
        
        return behavior
    
    @staticmethod
    def format_frame_results(
        frame_id: int,
        detections: List[Dict],
        confirmed_particles: List[FireParticle],
        monitoring_particles: List[FireParticle],
        resource_metrics: Dict[str, float],
        jetson_id: str,
        capture_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Format complete frame results for MQTT."""
        if capture_time is None:
            capture_time = time.time()
        
        send_time = time.time()
        
        return {
            "jetson_id": jetson_id,
            "frame_id": frame_id,
            "timestamps": {
                "capture_time": capture_time,
                "send_time": send_time
            },
            "detections": [MessageFormatter.format_detection(d) for d in detections],
            "confirmed_particles": [
                {
                    "particle": MessageFormatter.format_particle(p),
                    "physics_metrics": MessageFormatter.format_physics_metrics(p),
                    "behavior_analytics": MessageFormatter.format_behavior_analytics(p)
                }
                for p in confirmed_particles
            ],
            "monitoring_particles": [
                {
                    "particle": MessageFormatter.format_particle(p),
                    "physics_metrics": MessageFormatter.format_physics_metrics(p),
                    "behavior_analytics": MessageFormatter.format_behavior_analytics(p)
                }
                for p in monitoring_particles
            ],
            "resource_metrics": resource_metrics,
            "summary": {
                "num_detections": len(detections),
                "num_confirmed": len(confirmed_particles),
                "num_monitoring": len(monitoring_particles)
            }
        }
    
    @staticmethod
    def to_json(message: Dict[str, Any]) -> str:
        """Convert message dictionary to JSON string."""
        return json.dumps(message, default=str)
