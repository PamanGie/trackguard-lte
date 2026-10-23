"""
Fallen Memory for Physics-Based Prediction
==========================================

Memory system untuk fallen objects dengan physics prediction menggunakan
inertia dan momentum conservation.

Konsep:
- Motor jatuh → punya velocity [vx, vy]
- YOLO hilang (occlusion) → motor TETAP bergerak dengan velocity yang sama (inertia)
- Predict posisi berikutnya menggunakan physics
"""

import numpy as np
from typing import List, Dict, Optional


class FallenMemory:
    """
    Memory untuk fallen objects dengan physics prediction
    
    Menggunakan inertia (Newton's 1st Law) untuk memprediksi posisi
    objek yang jatuh meskipun YOLO tidak detect lagi.
    """
    
    def __init__(self, track_id: int, bbox: List[float], velocity: np.ndarray, 
                 frame_id: int, config: Dict):
        """
        Initialize fallen memory
        
        Args:
            track_id: Track ID dari fallen object
            bbox: Bounding box [x1, y1, x2, y2] saat pertama kali jatuh terdeteksi
            velocity: Velocity vector [vx, vy] dalam px/frame
            frame_id: Frame ID saat pertama kali jatuh terdeteksi
            config: Configuration dari PHYSICS_CONFIG['fallen_detector']
        """
        # Identity
        self.track_id = track_id
        
        # Physics state - dari memori awal kali terdeteksi jatuh
        self.position = np.array([bbox[0], bbox[1]], dtype=np.float32)  # Top-left corner
        self.velocity = velocity.copy().astype(np.float32)  # [vx, vy] dalam px/frame
        self.bbox_size = np.array([
            bbox[2] - bbox[0],  # width
            bbox[3] - bbox[1]   # height
        ], dtype=np.float32)
        
        # Inertia parameters dari config
        self.friction = config.get('friction_coefficient', 0.95)  # Decay factor
        self.max_predict_frames = config.get('max_prediction_frames', 60)  # Max 2 detik (30 fps)
        self.confidence_decay = config.get('confidence_decay_rate', 0.97)  # Confidence decay per frame
        self.min_confidence = config.get('min_prediction_confidence', 0.3)  # Min confidence untuk render
        
        # State tracking
        self.fallen_since_frame = frame_id
        self.last_seen_frame = frame_id
        self.confidence = 1.0  # Start dengan confidence penuh
        
        # Store original bbox untuk reference
        self.original_bbox = bbox.copy()
        
        # Physics-based termination tracking
        # Energy tracking untuk termination
        self.initial_energy = self._compute_kinetic_energy(velocity, bbox)
        self.low_energy_frames = 0  # Counter untuk frames dengan energy < 5%
        self.energy_history = [self.initial_energy]  # Track energy history
        
        # Vorticity untuk direction of fall
        self.vorticity = 0.0  # Will be computed from velocity field
        self.fall_direction = np.array([0.0, 0.0], dtype=np.float32)  # Direction vector
    
    def predict_next_position(self) -> List[float]:
        """
        Predict posisi berikutnya menggunakan inertia (Newton's 1st Law)
        
        Formula:
        - x(t+1) = x(t) + v(t) * dt
        - v(t+1) = v(t) * friction (motor jatuh perlahan berhenti karena gesekan)
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Update position: x(t+1) = x(t) + v(t) * dt
        # FIX: Pastikan velocity digunakan dengan benar untuk update position
        self.position += self.velocity
        
        # Apply friction (motor jatuh perlahan berhenti karena gesekan)
        self.velocity *= self.friction
        
        # Update energy tracking untuk termination
        current_energy = self._compute_kinetic_energy(self.velocity, self.get_predicted_bbox())
        self.energy_history.append(current_energy)
        
        # Check energy dissipation (untuk termination)
        if self.initial_energy > 1e-6:
            energy_ratio = current_energy / self.initial_energy
            if energy_ratio < 0.05:  # < 5% dari initial energy
                self.low_energy_frames += 1
            else:
                self.low_energy_frames = 0  # Reset counter
        
        # Decay confidence (semakin lama tidak terlihat, semakin tidak yakin)
        self.confidence *= self.confidence_decay
        
        return self.get_predicted_bbox()
    
    def get_predicted_bbox(self) -> List[float]:
        """
        Get predicted bounding box dari current position
        
        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        x1, y1 = int(self.position[0]), int(self.position[1])
        x2 = x1 + int(self.bbox_size[0])
        y2 = y1 + int(self.bbox_size[1])
        
        return [x1, y1, x2, y2]
    
    def update_with_detection(self, bbox: List[float], velocity: np.ndarray, frame_id: int, 
                             vorticity: float = 0.0):
        """
        Update memory jika YOLO deteksi lagi
        
        Args:
            bbox: New bounding box dari YOLO
            velocity: New velocity vector
            frame_id: Current frame ID
            vorticity: Vorticity value untuk direction of fall
        """
        # Update position dan velocity dengan detection baru
        self.position = np.array([bbox[0], bbox[1]], dtype=np.float32)
        self.velocity = velocity.copy().astype(np.float32)
        self.last_seen_frame = frame_id
        
        # Update vorticity dan fall direction
        self.vorticity = vorticity
        if np.linalg.norm(velocity) > 1e-6:
            # Normalize velocity untuk direction vector
            self.fall_direction = velocity / np.linalg.norm(velocity)
        else:
            # Jika velocity sangat kecil, keep previous direction
            if np.linalg.norm(self.fall_direction) < 1e-6:
                self.fall_direction = np.array([1.0, 0.0], dtype=np.float32)  # Default right
        
        # Update bbox size (jika berubah)
        new_width = bbox[2] - bbox[0]
        new_height = bbox[3] - bbox[1]
        # Average dengan size lama untuk smooth transition
        self.bbox_size = 0.7 * self.bbox_size + 0.3 * np.array([new_width, new_height], dtype=np.float32)
        
        # Update energy tracking
        current_energy = self._compute_kinetic_energy(velocity, bbox)
        self.energy_history.append(current_energy)
        # Keep only last 10 energy values
        if len(self.energy_history) > 10:
            self.energy_history = self.energy_history[-10:]
        
        # Reset confidence (YOLO detect lagi = confidence penuh)
        self.confidence = 1.0
        # Reset low energy counter
        self.low_energy_frames = 0
    
    def should_expire(self, current_frame: int) -> bool:
        """
        Check apakah memory sudah kadaluarsa
        
        Args:
            current_frame: Current frame ID
            
        Returns:
            True jika memory harus dihapus
        """
        frames_since_seen = current_frame - self.last_seen_frame
        
        # Expire jika:
        # 1. Sudah terlalu lama tidak terlihat (> max_predict_frames)
        # 2. Confidence terlalu rendah (< min_confidence)
        return (frames_since_seen > self.max_predict_frames or 
                self.confidence < self.min_confidence)
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity vector"""
        return self.velocity.copy()
    
    def get_confidence(self) -> float:
        """Get current confidence"""
        return self.confidence
    
    def get_frames_since_seen(self, current_frame: int) -> int:
        """Get number of frames since last seen"""
        return current_frame - self.last_seen_frame
    
    def _compute_kinetic_energy(self, velocity: np.ndarray, bbox: List[float]) -> float:
        """
        Compute kinetic energy: E = 0.5 * m * v^2
        
        Args:
            velocity: Velocity vector [vx, vy]
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Kinetic energy
        """
        # Mass proportional to bbox area
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        mass = width * height
        
        # Kinetic energy
        speed_squared = np.linalg.norm(velocity)**2
        energy = 0.5 * mass * speed_squared
        
        return float(energy)
    
    def should_terminate_energy(self, frames_threshold: int = 4) -> bool:
        """
        Check apakah harus terminate karena energy dissipation complete
        
        Args:
            frames_threshold: Number of consecutive frames dengan energy < 5%
            
        Returns:
            True jika harus terminate
        """
        return self.low_energy_frames >= frames_threshold
    
    def get_fall_direction(self) -> np.ndarray:
        """Get normalized fall direction vector"""
        return self.fall_direction.copy()
    
    def get_vorticity(self) -> float:
        """Get vorticity value"""
        return self.vorticity


# Testing
if __name__ == "__main__":
    print("Testing FallenMemory...")
    
    # Create config
    config = {
        'friction_coefficient': 0.95,
        'max_prediction_frames': 60,
        'confidence_decay_rate': 0.97,
        'min_prediction_confidence': 0.3
    }
    
    # Create fallen memory
    track_id = 1
    bbox = [100, 200, 150, 250]  # [x1, y1, x2, y2]
    velocity = np.array([10.0, 2.0])  # Bergerak ke kanan-bawah
    frame_id = 10
    
    memory = FallenMemory(track_id, bbox, velocity, frame_id, config)
    
    print(f"✓ FallenMemory created:")
    print(f"  Track ID: {memory.track_id}")
    print(f"  Initial position: {memory.position}")
    print(f"  Initial velocity: {memory.velocity}")
    print(f"  Bbox size: {memory.bbox_size}")
    
    # Test prediction
    print("\nTesting prediction:")
    for i in range(5):
        predicted_bbox = memory.predict_next_position()
        print(f"  Frame {frame_id + i + 1}: bbox = {predicted_bbox}, "
              f"velocity = {memory.velocity}, confidence = {memory.confidence:.3f}")
    
    # Test update
    print("\nTesting update with detection:")
    new_bbox = [160, 210, 210, 260]
    new_velocity = np.array([5.0, -15.0])  # Bergerak ke atas
    memory.update_with_detection(new_bbox, new_velocity, frame_id + 10)
    print(f"  Updated position: {memory.position}")
    print(f"  Updated velocity: {memory.velocity}")
    print(f"  Confidence reset: {memory.confidence}")
    
    # Test expiry
    print("\nTesting expiry:")
    should_expire = memory.should_expire(frame_id + 100)
    print(f"  Should expire (frame {frame_id + 100}): {should_expire}")
    
    print("\n✓ FallenMemory test completed")

