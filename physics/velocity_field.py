"""
Velocity Field and Shared Physics Calculations for LTE-TrackGuard
=================================================================

Provides shared physics computations untuk semua behaviour detectors:
- Velocity field construction (Blueprint Section 2.4.1, 2.6.4)
- Divergence calculation (for brake detection)
- Vorticity/Cross product (for turn detection)
- Streamline simulation (for physics predictor)

From Blueprint Sections 2.4.1, 2.6.4
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time


class VelocityField:
    """
    Velocity field constructor and physics calculator
    
    Shared component untuk semua physics detectors
    """
    
    def __init__(self, config: Dict):
        """
        Initialize velocity field calculator
        
        Args:
            config: Configuration dari PHYSICS_CONFIG['velocity_field']
        """
        self.config = config
        
        # Parameters dari blueprint
        self.gaussian_radius = config.get('gaussian_radius', 100.0)
        self.flow_sigma = config.get('flow_sigma', 50.0)
        
        # Cache untuk velocity field (per frame)
        self.cached_velocities = {}
        self.cached_field = None
        self.cache_frame_id = -1
        
        print("✓ VelocityField initialized")
        print(f"  Gaussian radius: {self.gaussian_radius} px")
        print(f"  Flow sigma: {self.flow_sigma} px")
    
    def compute_velocity(self, track: object, dt: float = 1.0) -> np.ndarray:
        """
        Compute velocity untuk single track (Blueprint Section 2.4.1)

        v_i(t) = [x_i(t) - x_i(t-1)] / Δt

        Args:
            track: Track object dengan history
            dt: Time delta (default 1 frame)

        Returns:
            Velocity vector [vx, vy] dalam pixels/frame
        """
        # Need at least 2 frames untuk velocity
        if not hasattr(track, 'history') or len(track.history) < 2:
            return np.array([0.0, 0.0])

        # Current dan previous positions
        current_center = track.current_detection.get('center', [0, 0])
        prev_center = track.history[-2].get('center', current_center)

        # Finite difference
        vx = (current_center[0] - prev_center[0]) / dt
        vy = (current_center[1] - prev_center[1]) / dt

        return np.array([vx, vy])

    def compute_velocity_smooth(self, track: object, window: int = 5) -> np.ndarray:
        """
        Smoothed velocity dari N-frame window — lebih robust terhadap YOLO jitter.

        Dipakai untuk motion_history direction tracking supaya angular velocity
        tidak collapse ke 0.0°/fr akibat jitter 1-2 frame.

        Cara: ambil centroid paling awal dan paling akhir dalam window,
        bagi dengan jumlah frame → velocity rata-rata yang stabil.
        Tetap responsif terhadap sudden stop (window pendek = 5 frame).

        Fallback ke compute_velocity() jika history < window.
        """
        if not hasattr(track, 'history') or len(track.history) < 2:
            return np.array([0.0, 0.0])

        hist = track.history
        n = min(window, len(hist))
        if n < 2:
            return self.compute_velocity(track)

        # Endpoint dari window
        recent = hist[-1].get('center', [0, 0])
        older  = hist[-n].get('center', recent)
        dt     = n - 1  # jumlah frame antar endpoint

        vx = (recent[0] - older[0]) / dt
        vy = (recent[1] - older[1]) / dt
        return np.array([vx, vy])
    
    def compute_divergence(self, track_i: object, neighbors: List[object], 
                          dt: float = 1.0, tau: int = 3) -> float:
        """
        Compute divergence untuk brake detection (Blueprint Section 2.4.3)
        
        Div_i(t) = [d_front,i(t) - d_front,i(t-τ)] / [Δt · d_front,i(t-τ)]
        
        Args:
            track_i: Target track
            neighbors: List of neighboring tracks (untuk find vehicle di depan)
            dt: Time delta
            tau: Temporal window (frames)
            
        Returns:
            Divergence value (negative = compression/braking)
        """
        # Find leading vehicle (kendaraan di depan)
        leading_vehicle = self._find_leading_vehicle(track_i, neighbors)
        
        if leading_vehicle is None:
            return 0.0  # No vehicle ahead, no divergence
        
        # Current distance
        d_current = self._compute_distance(track_i, leading_vehicle)
        
        # Distance tau frames ago
        if len(track_i.history) <= tau:
            return 0.0
        
        # Reconstruct past distance (approximate)
        past_center_i = track_i.history[-tau-1].get('center', track_i.current_detection['center'])
        
        # Cari leading vehicle position tau frames ago (approximate dengan current)
        d_past = self._compute_distance_from_centers(
            past_center_i, 
            leading_vehicle.current_detection['center']
        )
        
        # Avoid division by zero
        if d_past < 1e-6:
            return 0.0
        
        # Divergence calculation (Blueprint Eq. 16)
        divergence = (d_current - d_past) / (dt * tau * d_past)
        
        return divergence
    
    def compute_cross_product_2d(self, track: object) -> float:
        """
        Compute 2D cross product untuk turn detection (Blueprint Section 2.4.2)
        
        CP_i(t) = v_x,i(t-1) · v_y,i(t) - v_y,i(t-1) · v_x,i(t)
        
        Args:
            track: Track object dengan velocity history
            
        Returns:
            Cross product value (+ = right turn, - = left turn, 0 = straight)
        """
        # Need at least 2 velocity samples
        if not hasattr(track, 'history') or len(track.history) < 3:
            return 0.0
        
        # Compute current and previous velocities
        v_current = self.compute_velocity(track, dt=1.0)
        
        # Previous velocity (dari t-1 ke t-2)
        if len(track.history) >= 3:
            center_t_minus_1 = track.history[-2].get('center', [0, 0])
            center_t_minus_2 = track.history[-3].get('center', center_t_minus_1)
            
            v_prev = np.array([
                center_t_minus_1[0] - center_t_minus_2[0],
                center_t_minus_1[1] - center_t_minus_2[1]
            ])
        else:
            v_prev = v_current
        
        # 2D Cross product (Blueprint Eq. 14)
        cross_product = v_prev[0] * v_current[1] - v_prev[1] * v_current[0]
        
        return cross_product
    
    def build_global_field(self, tracks: List[object], frame_id: int) -> np.ndarray:
        """
        Build global velocity field dari active tracks (Blueprint Section 2.6.4)
        
        V(x) = Σ w_i(x) · v_i / Σ w_i(x)
        where w_i(x) = exp(-||x - x_i||² / 2r²)
        
        Args:
            tracks: List of active tracks
            frame_id: Current frame ID (untuk caching)
            
        Returns:
            Field info (untuk streamline simulation nanti)
        """
        # Check cache
        if frame_id == self.cache_frame_id and self.cached_field is not None:
            return self.cached_field
        
        # Store track positions and velocities
        field_data = {
            'positions': [],
            'velocities': [],
            'frame_id': frame_id
        }
        
        for track in tracks:
            if not hasattr(track, 'current_detection'):
                continue
            
            center = track.current_detection.get('center', [0, 0])
            velocity = self.compute_velocity(track)
            
            field_data['positions'].append(center)
            field_data['velocities'].append(velocity)
        
        # Convert to numpy arrays
        field_data['positions'] = np.array(field_data['positions'])
        field_data['velocities'] = np.array(field_data['velocities'])
        
        # Cache
        self.cached_field = field_data
        self.cache_frame_id = frame_id
        
        return field_data
    
    def sample_field_at_position(self, position: np.ndarray, 
                                 field_data: Dict) -> np.ndarray:
        """
        Sample velocity field at specific position (untuk streamline)
        
        Args:
            position: [x, y] position
            field_data: Field data dari build_global_field()
            
        Returns:
            Interpolated velocity [vx, vy]
        """
        if len(field_data['positions']) == 0:
            return np.array([0.0, 0.0])
        
        positions = field_data['positions']
        velocities = field_data['velocities']
        
        # Compute Gaussian weights (Blueprint Eq. 19)
        distances = np.linalg.norm(positions - position, axis=1)
        weights = np.exp(-distances**2 / (2 * self.gaussian_radius**2))
        
        # Normalize weights
        weight_sum = np.sum(weights)
        if weight_sum < 1e-6:
            return np.array([0.0, 0.0])
        
        weights = weights / weight_sum
        
        # Weighted average velocity
        v_interpolated = np.sum(velocities * weights[:, np.newaxis], axis=0)
        
        return v_interpolated
    
    def _find_leading_vehicle(self, track_i: object, neighbors: List[object], 
                              max_distance: float = 100.0) -> Optional[object]:
        """
        Find kendaraan di depan track_i (untuk divergence calculation)
        
        Args:
            track_i: Target track
            neighbors: List of potential leading vehicles
            max_distance: Maximum search distance
            
        Returns:
            Leading vehicle track atau None
        """
        center_i = track_i.current_detection.get('center', [0, 0])
        velocity_i = self.compute_velocity(track_i)
        
        # Direction of motion (normalized)
        speed_i = np.linalg.norm(velocity_i)
        if speed_i < 1e-6:
            return None
        
        direction_i = velocity_i / speed_i
        
        # Find vehicle in front (same lane, ahead in motion direction)
        candidates = []
        
        for neighbor in neighbors:
            if neighbor.track_id == track_i.track_id:
                continue
            
            center_j = neighbor.current_detection.get('center', [0, 0])
            
            # Vector from i to j
            vec_ij = np.array([center_j[0] - center_i[0], center_j[1] - center_i[1]])
            distance = np.linalg.norm(vec_ij)
            
            if distance > max_distance or distance < 1e-6:
                continue
            
            # Check if j is in front (dot product > 0)
            direction_ij = vec_ij / distance
            alignment = np.dot(direction_i, direction_ij)
            
            if alignment > 0.5:  # Same direction, ahead
                # Check y-coordinate proximity (same lane)
                y_diff = abs(center_j[1] - center_i[1])
                if y_diff < 50:  # Within 50 pixels vertically (same lane)
                    candidates.append((neighbor, distance))
        
        # Return closest candidate
        if len(candidates) == 0:
            return None
        
        candidates.sort(key=lambda x: x[1])  # Sort by distance
        return candidates[0][0]
    
    def _compute_distance(self, track_i: object, track_j: object) -> float:
        """Compute Euclidean distance between two tracks"""
        center_i = track_i.current_detection.get('center', [0, 0])
        center_j = track_j.current_detection.get('center', [0, 0])
        
        return self._compute_distance_from_centers(center_i, center_j)
    
    def _compute_distance_from_centers(self, center_i: List, center_j: List) -> float:
        """Compute distance from center coordinates"""
        return np.sqrt((center_i[0] - center_j[0])**2 + (center_i[1] - center_j[1])**2)
    
    def get_statistics(self) -> Dict:
        """Get velocity field statistics"""
        return {
            'gaussian_radius': self.gaussian_radius,
            'flow_sigma': self.flow_sigma,
            'cache_hits': 1 if self.cached_field is not None else 0,
            'cached_tracks': len(self.cached_field.get('positions', [])) if self.cached_field else 0
        }


# Testing
if __name__ == "__main__":
    print("Testing VelocityField...")
    
    # Create velocity field
    config = {'gaussian_radius': 100.0, 'flow_sigma': 50.0}
    vf = VelocityField(config)
    
    # Create dummy track
    class DummyTrack:
        def __init__(self, track_id, centers):
            self.track_id = track_id
            self.history = [{'center': c} for c in centers]
            self.current_detection = {'center': centers[-1]}
    
    track = DummyTrack(1, [[100, 100], [110, 105], [120, 110]])
    
    # Test velocity computation
    velocity = vf.compute_velocity(track)
    print(f"✓ Velocity computed: {velocity}")
    
    # Test cross product
    cp = vf.compute_cross_product_2d(track)
    print(f"✓ Cross product: {cp:.2f}")
    
    # Test global field
    tracks = [track]
    field = vf.build_global_field(tracks, frame_id=1)
    print(f"✓ Global field built: {len(field['positions'])} tracks")
    
    # Test field sampling
    sampled_v = vf.sample_field_at_position(np.array([115, 107]), field)
    print(f"✓ Field sampled: {sampled_v}")
    
    print("✓ VelocityField test completed")
