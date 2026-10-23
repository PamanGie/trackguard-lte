"""
Scene Analyzer for LTE-TrackGuard
==================================

Analyze traffic density dan adjust physics detector thresholds adaptively.

Traffic Density Classification:
- Sparse: < 5 tracks/Mpx (night, empty highway)
- Normal: 5-15 tracks/Mpx (regular traffic)
- Dense: 15-30 tracks/Mpx (rush hour)
- Congested: > 30 tracks/Mpx (traffic jam)

From Blueprint Section 2.7: Adaptive Threshold berdasarkan Scene Context
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class SceneAnalyzer:
    """
    Scene analyzer untuk traffic density classification dan adaptive thresholding
    
    Adjusts behaviour detector thresholds based on current traffic conditions
    """
    
    def __init__(self, config: Dict):
        """
        Initialize scene analyzer
        
        Args:
            config: Configuration dari PHYSICS_CONFIG['scene_analyzer']
        """
        self.config = config
        
        # Density thresholds dari blueprint (tracks per megapixel)
        self.density_sparse = config.get('density_sparse', 5.0)
        self.density_normal = config.get('density_normal', 15.0)
        self.density_dense = config.get('density_dense', 30.0)
        
        # Adaptive weights per density level
        self.adaptive_weights = config.get('adaptive_weights', {
            'sparse': {'tau_brake': -0.5, 'tau_turn': 0.3},
            'normal': {'tau_brake': -0.8, 'tau_turn': 0.5},
            'dense': {'tau_brake': -1.2, 'tau_turn': 0.7},
            'congested': {'tau_brake': -1.5, 'tau_turn': 0.9}
        })
        
        # Scene state tracking
        self.current_density = 0.0
        self.current_category = 'normal'
        self.density_history = []
        self.category_history = []
        
        # Frame dimensions (updated on first analyze call)
        self.frame_width = None
        self.frame_height = None
        self.frame_area_mpx = None
        
        # Statistics
        self.total_analyses = 0
        self.category_counts = {
            'sparse': 0,
            'normal': 0,
            'dense': 0,
            'congested': 0
        }
        
        print("✓ SceneAnalyzer initialized")
        print(f"  Density thresholds: sparse<{self.density_sparse}, "
              f"normal<{self.density_normal}, dense<{self.density_dense}")
    
    def analyze_scene(self, tracks: List, frame_shape: Tuple[int, int]) -> Dict:
        """
        Analyze current scene dan return adaptive parameters
        
        Args:
            tracks: List of active tracks
            frame_shape: (height, width) dari video frame
            
        Returns:
            Scene analysis result dengan adaptive parameters
        """
        self.total_analyses += 1
        
        # Update frame dimensions jika first call atau changed
        if (self.frame_width is None or 
            self.frame_width != frame_shape[1] or 
            self.frame_height != frame_shape[0]):
            
            self.frame_width = frame_shape[1]
            self.frame_height = frame_shape[0]
            self.frame_area_mpx = (self.frame_width * self.frame_height) / 1e6
            
            print(f"  Scene dimensions: {self.frame_width}x{self.frame_height} "
                  f"({self.frame_area_mpx:.2f} Mpx)")
        
        # Calculate traffic density (Blueprint Section 2.7)
        num_tracks = len(tracks)
        density = num_tracks / self.frame_area_mpx if self.frame_area_mpx > 0 else 0
        
        # Classify density category
        category = self._classify_density(density)
        
        # Update state
        self.current_density = density
        self.current_category = category
        self.density_history.append(density)
        self.category_history.append(category)
        
        # Keep history limited
        if len(self.density_history) > 100:
            self.density_history = self.density_history[-100:]
            self.category_history = self.category_history[-100:]
        
        # Update statistics
        self.category_counts[category] = self.category_counts.get(category, 0) + 1
        
        # Get adaptive parameters
        adaptive_params = self._get_adaptive_parameters(category)
        
        # Additional scene metrics
        scene_metrics = self._compute_scene_metrics(tracks)
        
        return {
            'density': density,
            'category': category,
            'num_tracks': num_tracks,
            'adaptive_params': adaptive_params,
            'scene_metrics': scene_metrics,
            'frame_area_mpx': self.frame_area_mpx
        }
    
    def _classify_density(self, density: float) -> str:
        """
        Classify traffic density category (Blueprint Section 2.7)
        
        Args:
            density: Tracks per megapixel
            
        Returns:
            Category: 'sparse', 'normal', 'dense', or 'congested'
        """
        if density < self.density_sparse:
            return 'sparse'
        elif density < self.density_normal:
            return 'normal'
        elif density < self.density_dense:
            return 'dense'
        else:
            return 'congested'
    
    def _get_adaptive_parameters(self, category: str) -> Dict:
        """
        Get adaptive parameters untuk category (Blueprint Section 2.7)
        
        Args:
            category: Traffic density category
            
        Returns:
            Dict of adaptive parameters untuk behaviour detectors
        """
        # Base parameters dari config
        base_params = self.adaptive_weights.get(category, 
                                                self.adaptive_weights['normal'])
        
        # Additional adaptive rules
        adaptive_params = {
            # Brake detector
            'brake': {
                'tau_brake': base_params.get('tau_brake', -0.8),
                'explanation': f'Relaxed for {category} traffic'
            },
            
            # Turn detector
            'turn': {
                'tau_turn': base_params.get('tau_turn', 0.5),
                'explanation': f'Adjusted for {category} traffic'
            },
            
            # Fallen detector (less affected by density)
            'fallen': {
                'persistence_multiplier': 1.0,  # Keep strict
                'explanation': 'Independent of density'
            },
            
            # Collision detector
            'collision': {
                'iou_threshold_adjustment': 0.0 if category != 'congested' else 0.1,
                'explanation': f'Higher IoU tolerance for {category}' if category == 'congested' else 'Normal'
            }
        }
        
        return adaptive_params
    
    def _compute_scene_metrics(self, tracks: List) -> Dict:
        """
        Compute additional scene metrics
        
        Args:
            tracks: List of tracks
            
        Returns:
            Scene metrics dict
        """
        if len(tracks) == 0:
            return {
                'avg_velocity': 0.0,
                'velocity_std': 0.0,
                'avg_track_age': 0.0,
                'spatial_distribution': 'unknown'
            }
        
        # Average velocity
        velocities = []
        ages = []
        positions = []
        
        for track in tracks:
            # Velocity
            if hasattr(track, 'history') and len(track.history) >= 2:
                curr = track.current_detection.get('center', [0, 0])
                prev = track.history[-2].get('center', curr)
                v = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                velocities.append(v)
            
            # Age
            if hasattr(track, 'age'):
                ages.append(track.age)
            
            # Position
            positions.append(track.current_detection.get('center', [0, 0]))
        
        avg_velocity = np.mean(velocities) if velocities else 0.0
        velocity_std = np.std(velocities) if velocities else 0.0
        avg_age = np.mean(ages) if ages else 0.0
        
        # Spatial distribution (simple: check clustering)
        spatial_dist = self._analyze_spatial_distribution(positions)
        
        return {
            'avg_velocity': float(avg_velocity),
            'velocity_std': float(velocity_std),
            'avg_track_age': float(avg_age),
            'spatial_distribution': spatial_dist
        }
    
    def _analyze_spatial_distribution(self, positions: List[List[float]]) -> str:
        """
        Analyze spatial distribution of tracks
        
        Args:
            positions: List of [x, y] positions
            
        Returns:
            Distribution type: 'clustered', 'uniform', or 'sparse'
        """
        if len(positions) < 3:
            return 'sparse'
        
        positions = np.array(positions)
        
        # Compute pairwise distances
        distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        if len(distances) == 0:
            return 'sparse'
        
        # Coefficient of variation
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        if mean_dist < 1e-6:
            return 'clustered'
        
        cv = std_dist / mean_dist
        
        # Classification based on CV
        if cv < 0.5:
            return 'uniform'  # Low variation = evenly distributed
        elif cv < 1.0:
            return 'mixed'
        else:
            return 'clustered'  # High variation = clustered
    
    def apply_adaptive_thresholds(self, behaviour_detectors: Dict, 
                                  adaptive_params: Dict) -> Dict:
        """
        Apply adaptive thresholds ke behaviour detectors
        
        Args:
            behaviour_detectors: Dict of detector instances
            adaptive_params: Adaptive parameters dari analyze_scene
            
        Returns:
            Original thresholds (untuk restore nanti)
        """
        original_thresholds = {}
        
        # Brake detector
        if 'brake' in behaviour_detectors and 'brake' in adaptive_params:
            detector = behaviour_detectors['brake']
            original_thresholds['brake'] = {
                'tau_brake': detector.tau_brake
            }
            
            # Apply adaptive threshold
            new_tau = adaptive_params['brake']['tau_brake']
            detector.tau_brake = new_tau
        
        # Turn detector
        if 'turn' in behaviour_detectors and 'turn' in adaptive_params:
            detector = behaviour_detectors['turn']
            original_thresholds['turn'] = {
                'tau_turn_right': detector.tau_turn_right,
                'tau_turn_left': detector.tau_turn_left
            }
            
            # Apply adaptive threshold
            new_tau = adaptive_params['turn']['tau_turn']
            detector.tau_turn_right = new_tau
            detector.tau_turn_left = -new_tau
        
        # Collision detector
        if 'collision' in behaviour_detectors and 'collision' in adaptive_params:
            detector = behaviour_detectors['collision']
            original_thresholds['collision'] = {
                'iou_overlap_threshold': detector.iou_overlap_threshold
            }
            
            # Apply adaptive adjustment
            adjustment = adaptive_params['collision']['iou_threshold_adjustment']
            detector.iou_overlap_threshold += adjustment
        
        return original_thresholds
    
    def restore_thresholds(self, behaviour_detectors: Dict, 
                          original_thresholds: Dict):
        """
        Restore original thresholds setelah adaptive application
        
        Args:
            behaviour_detectors: Dict of detector instances
            original_thresholds: Original thresholds dari apply_adaptive_thresholds
        """
        # Brake
        if 'brake' in original_thresholds and 'brake' in behaviour_detectors:
            detector = behaviour_detectors['brake']
            detector.tau_brake = original_thresholds['brake']['tau_brake']
        
        # Turn
        if 'turn' in original_thresholds and 'turn' in behaviour_detectors:
            detector = behaviour_detectors['turn']
            detector.tau_turn_right = original_thresholds['turn']['tau_turn_right']
            detector.tau_turn_left = original_thresholds['turn']['tau_turn_left']
        
        # Collision
        if 'collision' in original_thresholds and 'collision' in behaviour_detectors:
            detector = behaviour_detectors['collision']
            detector.iou_overlap_threshold = original_thresholds['collision']['iou_overlap_threshold']
    
    def get_statistics(self) -> Dict:
        """Get scene analyzer statistics"""
        return {
            'total_analyses': self.total_analyses,
            'current_density': self.current_density,
            'current_category': self.current_category,
            'category_distribution': self.category_counts,
            'frame_dimensions': f'{self.frame_width}x{self.frame_height}' if self.frame_width else 'unknown',
            'frame_area_mpx': self.frame_area_mpx
        }
    
    def get_density_trend(self, window: int = 30) -> Dict:
        """
        Get traffic density trend over recent frames
        
        Args:
            window: Window size untuk trend analysis
            
        Returns:
            Trend analysis dict
        """
        if len(self.density_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_densities = self.density_history[-window:]
        
        if len(recent_densities) < 5:
            return {'trend': 'insufficient_data'}
        
        # Simple linear regression untuk trend
        x = np.arange(len(recent_densities))
        y = np.array(recent_densities)
        
        # Slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Classify trend
        if abs(slope) < 0.1:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing' if slope > 0.5 else 'slightly_increasing'
        else:
            trend = 'decreasing' if slope < -0.5 else 'slightly_decreasing'
        
        return {
            'trend': trend,
            'slope': float(slope),
            'current_density': float(recent_densities[-1]),
            'avg_density': float(np.mean(recent_densities)),
            'window_size': len(recent_densities)
        }


# Testing
if __name__ == "__main__":
    print("Testing SceneAnalyzer...")
    
    # Create analyzer
    config = {
        'density_sparse': 5.0,
        'density_normal': 15.0,
        'density_dense': 30.0,
        'adaptive_weights': {
            'sparse': {'tau_brake': -0.5, 'tau_turn': 0.3},
            'normal': {'tau_brake': -0.8, 'tau_turn': 0.5},
            'dense': {'tau_brake': -1.2, 'tau_turn': 0.7},
            'congested': {'tau_brake': -1.5, 'tau_turn': 0.9}
        }
    }
    
    analyzer = SceneAnalyzer(config)
    
    # Create dummy tracks
    class DummyTrack:
        def __init__(self, track_id, center):
            self.track_id = track_id
            self.age = 10
            self.history = [
                {'center': [center[0]-5, center[1]]},
                {'center': center}
            ]
            self.current_detection = {'center': center}
    
    # Test different density scenarios
    frame_shape = (1080, 1920)  # Full HD
    
    print("\nTesting density classifications:")
    
    # Sparse (2 tracks)
    sparse_tracks = [DummyTrack(i, [100+i*100, 100]) for i in range(2)]
    result_sparse = analyzer.analyze_scene(sparse_tracks, frame_shape)
    print(f"\nSparse scene:")
    print(f"  Tracks: {result_sparse['num_tracks']}")
    print(f"  Density: {result_sparse['density']:.2f} tracks/Mpx")
    print(f"  Category: {result_sparse['category']}")
    print(f"  Adaptive tau_brake: {result_sparse['adaptive_params']['brake']['tau_brake']}")
    
    # Normal (15 tracks)
    normal_tracks = [DummyTrack(i, [100+i*50, 100+i*30]) for i in range(15)]
    result_normal = analyzer.analyze_scene(normal_tracks, frame_shape)
    print(f"\nNormal scene:")
    print(f"  Tracks: {result_normal['num_tracks']}")
    print(f"  Density: {result_normal['density']:.2f} tracks/Mpx")
    print(f"  Category: {result_normal['category']}")
    print(f"  Adaptive tau_brake: {result_normal['adaptive_params']['brake']['tau_brake']}")
    
    # Dense (40 tracks)
    dense_tracks = [DummyTrack(i, [100+i*40, 100+i*20]) for i in range(40)]
    result_dense = analyzer.analyze_scene(dense_tracks, frame_shape)
    print(f"\nDense scene:")
    print(f"  Tracks: {result_dense['num_tracks']}")
    print(f"  Density: {result_dense['density']:.2f} tracks/Mpx")
    print(f"  Category: {result_dense['category']}")
    print(f"  Adaptive tau_brake: {result_dense['adaptive_params']['brake']['tau_brake']}")
    
    # Trend analysis
    trend = analyzer.get_density_trend(window=3)
    print(f"\nDensity trend:")
    print(f"  Trend: {trend['trend']}")
    print(f"  Slope: {trend.get('slope', 0):.3f}")
    
    # Statistics
    stats = analyzer.get_statistics()
    print(f"\n✓ SceneAnalyzer Statistics:")
    print(f"  Total analyses: {stats['total_analyses']}")
    print(f"  Category distribution: {stats['category_distribution']}")
    print(f"  Frame dimensions: {stats['frame_dimensions']}")
    
    print("\n✓ SceneAnalyzer test completed")
