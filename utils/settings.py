"""
LTE-TrackGuard Settings Configuration
======================================

Centralized settings untuk physics-based traffic behaviour detection.
Simplified version - no MOT17 dependencies.
"""

import os
import torch
from typing import Dict, Any


class Settings:
    """
    Centralized settings untuk LTE-TrackGuard
    """
    
    # =====================================
    # USER CONFIGURATION - MODIFY THESE
    # =====================================

    # Model Weights Configuration
    DETECTOR_WEIGHTS = "best_visdrone_full.pt"  # YOLOv11 Nano - fast & efficient untuk traffic
    # Alternatives:
    # - "yolo11s.pt" (small - lebih akurat, sedikit lebih lambat)
    # - "yolo11m.pt" (medium - balance)
    # - "yolo11l.pt" (large - high accuracy)
    # - "yolo11x.pt" (extra large - paling akurat, paling lambat)

    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Performance Preset
    PERFORMANCE_PRESET = "balanced"  # Options: "high_accuracy", "balanced", "high_recall"
    
    # =====================================
    # DETECTOR CONFIGURATION
    # =====================================
    
    DETECTOR_CONFIG = {
        'model_variant': 'yolo11n',  # Will be overridden by DETECTOR_WEIGHTS
        'confidence_threshold': 0.25,
        'nms_threshold': 0.45,
        'device': DEVICE,
        'input_size': 640,
        'half_precision': True if DEVICE == 'cuda' else False,
        
        # Multi-class detection for traffic (LTE-TrackGuard)
        'target_classes': ['car', 'motorcycle', 'truck', 'bus', 'person'],
        'class_ids': [2, 3, 7, 5, 0],  # COCO class IDs
        
        # Class-specific minimum sizes (width x height in pixels)
        'min_bbox_sizes': {
            'person': (8, 15),
            'car': (15, 15),
            'motorcycle': (15, 15),
            'truck': (15, 15),
            'bus': (15, 15)
        }
    }
    
    # =====================================
    # TRACKING CONFIGURATION
    # =====================================
    
    TRACKING_CONFIG = {
        # Smart Hungarian Algorithm Parameters
        'use_smart_hungarian': True,
        'iou_threshold': 0.3,
        'quality_threshold': 0.6,
        'max_age': 30,
        'min_hits': 3,
        
        # Ghost Node Configuration
        'enable_ghost_nodes': True,
        'ghost_max_age': 15,
        'ghost_iou_threshold': 0.3,
        
        # Spatial constraints
        'max_distance': 200.0,
        'occlusion_handling': True
    }
    
    # =====================================
    # PHYSICS CONFIGURATION (LTE-TrackGuard)
    # =====================================
    
    PHYSICS_CONFIG = {
        # Module switches
        'enable_eager': False,                    # EAGER smoothing
        'enable_physics_predictor': False,        # Physics predictor (EXPERIMENTAL)
        'enable_behaviour_detection': False,      # Behaviour detectors
        
        # EAGER Smoothing Parameters (Section 2.3)
        'eager': {
            'alpha': 0.2,           # Smoothing factor untuk traffic
            'w_internal': 0.3,      # Weight internal energy (AR consistency)
            'w_external': 0.8,      # Weight external energy (follow detection)
            'k_temp': 0.5,          # Weight temporal energy (jitter minimization)
            'iterations': 5         # Gradient descent iterations
        },
        
        # Velocity Field Parameters (Section 2.4.1, 2.6.4)
        'velocity_field': {
            'gaussian_radius': 100.0,   # Radius untuk velocity field weight
            'flow_sigma': 50.0          # Sigma untuk streamline fit
        },
        
        # Physics Predictor Parameters (Section 2.6) - EXPERIMENTAL
        'physics_predictor': {
            'sigma_momentum': 100.0,        # Tolerance momentum matching
            'sigma_energy': 500.0,          # Tolerance energy matching
            'w_iou': 0.35,                  # Weight IoU cost
            'w_momentum': 0.25,             # Weight momentum score
            'w_flow': 0.25,                 # Weight flow continuity
            'w_energy': 0.15,               # Weight energy signature
            'max_angle_diff': 45.0,         # Max angle change (degrees)
            'energy_ratio_min': 0.5,        # Min energy ratio
            'energy_ratio_max': 2.0,        # Max energy ratio
            'streamline_fit_max': 50.0      # Max streamline deviation (pixels)
        },
        
        # Turn Detector Parameters (Section 2.4.2)
        'turn_detector': {
            'tau_turn_right': 0.5,          # CP > 0.5 → belok kanan
            'tau_turn_left': -0.5,          # CP < -0.5 → belok kiri
            'tau_neutral': 0.15,            # |CP| ≤ 0.15 → lurus
            'window_size': 5,               # Temporal window (frames)
            'persist_threshold': 0.6,       # 60% persistence untuk konfirmasi
            'detect_zigzag': True           # Enable zig-zag detection
        },
        
        # Brake Detector Parameters (Section 2.4.3)
        'brake_detector': {
            'tau_brake': -0.8,              # Divergence threshold
            'tau_decel': 5.0,               # Deceleration threshold (px/frame)
            'window_size': 5,
            'persist_threshold': 0.6,
            # Adaptive thresholds based on speed
            'tau_brake_slow': -1.2,         # Speed < 10 px/frame
            'tau_brake_normal': -0.8,       # 10 ≤ speed < 30
            'tau_brake_fast': -0.5          # Speed ≥ 30
        },
        
        # Fallen Detector Parameters (Section 2.4.4)
        # SIMPLIFIED untuk video pendek - langsung alarm jika terdeteksi
        'fallen_detector': {
            'rotation_threshold': 50.0,     # Min rotation angle (degrees) - lebih rendah
            'ar_standing_min': 1.0,         # AR > 1.0 → standing (lebih rendah)
            'ar_fallen_max': 0.9,           # AR < 0.9 → fallen (lebih tinggi)
            'velocity_drop_ratio': 0.3,     # Sudden stop < 30% velocity (lebih longgar)
            'window_size': 3,               # Kurangi dari 5 → 3 frames
            'persist_threshold': 0.5,       # 50% persistence (simple - 2 dari 3 frames)
            
            # Physics prediction parameters (inertia & momentum)
            'enable_physics_prediction': True,  # Enable physics-based prediction untuk occlusion
            'friction_coefficient': 0.95,        # Motor jatuh friction (0.9-0.98)
            'max_prediction_frames': 60,       # Max 2 detik prediction (30 fps)
            'confidence_decay_rate': 0.97,     # Confidence decay per frame
            'min_prediction_confidence': 0.3    # Min confidence untuk render predicted bbox
        },
        
        # Collision Detector Parameters (Section 2.4.5)
        'collision_detector': {
            'iou_overlap_threshold': 0.3,   # Min IoU untuk collision
            'energy_loss_threshold': 0.9,   # Min energy loss (90% - Blueprint strict)
            'variance_acceleration': 5.0,   # Sudden acceleration variance (Blueprint: 5.0)
            'energy_loss_frames_back': 5,    # Frames to look back for energy loss (Blueprint: 5)
            'variance_frames_required': 4,   # Min frames required for variance
            'window_size': 5,
            'persist_threshold': 0.8,
            'persist_window': 1,  # 1 frame untuk deteksi instan
            # Physics-based collision detection (NEW - untuk deteksi berbasis gaya tabrakan)
            'impulse_threshold': 50.0,  # Minimum impulse magnitude (px²/frame)
            'impulse_frames_back': 2,  # Frames to look back for velocity change
            'acceleration_spike_threshold': 10.0,  # Minimum acceleration spike (px/frame²)
            'acceleration_spike_frames_back': 2,  # Frames to check for acceleration spike
            'force_threshold': 100.0,  # Minimum force magnitude (px²/frame²)
            # Deformation detection (NEW - untuk deteksi mobil ringsek)
            'ar_change_threshold': 0.12,  # 12% perubahan AR dari baseline (relaxed dari 20%)
            'area_change_threshold': 0.10,  # 10% perubahan area dari baseline (relaxed dari 15%)
            'baseline_frames': 10,  # Frames untuk baseline calculation
            # Multi-Tier Detection Strategy (untuk handle rear-end collision tanpa deformasi jelas)
            # Tier 1: High-Confidence (IoU > 0.5 + Energy > 0.8 + (Variance > 5.0 OR Rotation Spike > 0.8))
            # Rotation Spike: Spin tiba-tiba tanpa alasan = collision (mobil belok normal tidak akan spin)
            'tier1_iou_threshold': 0.5,  # IoU > 0.5 untuk high-confidence
            'tier1_energy_threshold': 0.8,  # Energy loss > 0.8 untuk high-confidence
            'tier1_variance_threshold': 5.0,  # Variance > 5.0 untuk high-confidence
            'tier1_rotation_spike_threshold': 0.8,  # Rotation spike > 0.8 untuk sudden spin (collision)
            # Tier 1.5: Sparse Scene High-Confidence (untuk scene sparse, bisa lebih agresif)
            'tier1_5_sparse_density_threshold': 1.0,  # Scene sparse jika < 1.0 tr/Mpx
            'tier1_5_iou_threshold': 0.4,  # IoU > 0.4 untuk sparse scene
            'tier1_5_energy_threshold': 0.7,  # Energy loss > 0.7 untuk sparse scene
            'tier1_5_rotation_spike_threshold': 0.6,  # Rotation spike > 0.6 untuk sparse scene
            # Push Collision Detection (Rear-End Collision) - NEW - PRIORITY TIER
            # Tier 0.5: Push Collision Priority (diprioritaskan pertama, threshold lebih rendah)
            'tier0_5_push_iou_threshold': 0.15,  # IoU > 0.15 untuk push collision (sangat agresif)
            'tier0_5_push_energy_threshold': 0.3,  # Energy loss > 0.3 untuk push collision (sangat agresif)
            'momentum_transfer_threshold': 10.0,  # Momentum transfer > 10 px²/frame (sangat relaxed)
            'relative_velocity_drop_threshold': 0.2,  # Relative velocity drop > 20% (sangat relaxed)
            'push_acceleration_threshold': 2.0,  # Acceleration spike > 2 px/frame² (sangat relaxed)
            'velocity_direction_change_threshold': 15.0,  # Direction change > 15° (sangat relaxed)
            'push_frames_back': 2,  # Frames back untuk compute push metrics
            # Traffic Jam Filter (untuk avoid false alarm di traffic padat merayap)
            'traffic_jam_velocity_threshold': 2.0,  # Velocity < 2 px/frame = traffic jam (merayap)
            'traffic_jam_velocity_change_threshold': 0.2,  # Velocity change ratio < 0.2 = konstan (traffic jam)
            'traffic_jam_energy_loss_threshold': 0.2,  # Energy loss < 0.2 = tidak ada loss (traffic jam)
            'traffic_jam_acceleration_variance_threshold': 2.0,  # Acceleration variance < 2.0 = gradual (traffic jam)
            # Proximity Warning (untuk objek mendekat, belum collision)
            'proximity_iou_threshold': 0.1,  # IoU > 0.1 untuk proximity warning (objek mendekat)
            'proximity_velocity_threshold': 3.0,  # Velocity > 3 px/frame untuk proximity (objek bergerak, bukan statis)
            'proximity_frames_back': 5,  # Frames back untuk check velocity pattern
            # Tier 2: Deformation-Based (IoU > 0.3 + Energy > 0.9 + Deformation) - perlu deformation
            'tier2_iou_threshold': 0.3,  # IoU > 0.3 untuk deformation-based
            'tier2_energy_threshold': 0.9,  # Energy loss > 0.9 untuk deformation-based (Blueprint strict)
            # Tier 3: Medium-Confidence (IoU > 0.4 + Energy > 0.7 + Variance > 3.0) - fallback
            'tier3_iou_threshold': 0.4,  # IoU > 0.4 untuk medium-confidence
            'tier3_energy_threshold': 0.7,  # Energy loss > 0.7 untuk medium-confidence
            'tier3_variance_threshold': 3.0,  # Variance > 3.0 untuk medium-confidence
            # Episode-level warning-to-escalation gate.
            # Yellow warning is emitted immediately; red escalation is raised
            # only after a short pair episode accumulates strong ARS while KELR
            # remains below the saturated nuisance-warning pattern.
            'episode_escalation': {
                'enabled': True,
                'ars_threshold': 2.9,
                'kelr_max_threshold': 0.89,
                'gap_frames': 30
            },
            # Conservative auto-escalation for behavioural/evasive warnings.
            # Mined from ALL_DETECTION logs: strong direction discontinuity plus
            # small but non-zero contact evidence can promote an orange critical
            # interaction to red escalation without promoting every warning.
            'evasive_escalation': {
                'enabled': True,
                'direction_change_threshold': 135.0,
                'iou_threshold': 0.004,
                'kelr_max_threshold': 0.95
            }
        },
        
        # Scene Analyzer Parameters (Section 2.7)
        'scene_analyzer': {
            # Traffic density thresholds (tracks per megapixel)
            'density_sparse': 5.0,
            'density_normal': 15.0,
            'density_dense': 30.0,
            # Adaptive weights per density
            'adaptive_weights': {
                'sparse': {'tau_brake': -0.5, 'tau_turn': 0.3},
                'normal': {'tau_brake': -0.8, 'tau_turn': 0.5},
                'dense': {'tau_brake': -1.2, 'tau_turn': 0.7},
                'congested': {'tau_brake': -1.5, 'tau_turn': 0.9}
            }
        }
    }
    
    # =====================================
    # INITIALIZATION
    # =====================================
    
    def __init__(self):
        """Initialize settings"""
        print(f"✓ Settings initialized")
        print(f"   Device: {self.DEVICE}")
        print(f"   YOLO Model: {self.DETECTOR_WEIGHTS}")
        print(f"   Performance Preset: {self.PERFORMANCE_PRESET}")
    
    # =====================================
    # GETTER METHODS
    # =====================================
    
    def get_detector_config(self) -> Dict[str, Any]:
        """Get detector configuration.

        Selalu sync model_variant dari DETECTOR_WEIGHTS agar --model arg
        di main.py langsung tercermin di sini tanpa perlu update dua tempat.
        """
        cfg = self.DETECTOR_CONFIG.copy()
        cfg['model_variant'] = self.DETECTOR_WEIGHTS  # weight file = source of truth
        return cfg
    
    def get_tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration"""
        return self.TRACKING_CONFIG.copy()
    
    def get_physics_config(self) -> Dict[str, Any]:
        """Get physics configuration"""
        return self.PHYSICS_CONFIG.copy()
    
    def get_device(self) -> str:
        """Get computation device"""
        return self.DEVICE
    
    def get_detector_weights_path(self) -> str:
        """Get detector weights path"""
        return self.DETECTOR_WEIGHTS
    
    # =====================================
    # PRESET CONFIGURATIONS
    # =====================================
    
    def apply_preset(self, preset: str):
        """
        Apply performance preset
        
        Args:
            preset: 'high_accuracy', 'balanced', or 'high_recall'
        """
        if preset == 'high_accuracy':
            self.DETECTOR_CONFIG['confidence_threshold'] = 0.4
            self.DETECTOR_CONFIG['nms_threshold'] = 0.3
            self.TRACKING_CONFIG['min_hits'] = 5
            self.TRACKING_CONFIG['quality_threshold'] = 0.7
            
        elif preset == 'balanced':
            self.DETECTOR_CONFIG['confidence_threshold'] = 0.25
            self.DETECTOR_CONFIG['nms_threshold'] = 0.45
            self.TRACKING_CONFIG['min_hits'] = 3
            self.TRACKING_CONFIG['quality_threshold'] = 0.6
            
        elif preset == 'high_recall':
            self.DETECTOR_CONFIG['confidence_threshold'] = 0.15
            self.DETECTOR_CONFIG['nms_threshold'] = 0.6
            self.TRACKING_CONFIG['min_hits'] = 2
            self.TRACKING_CONFIG['quality_threshold'] = 0.5
        
        print(f"✓ Applied preset: {preset}")


# Global settings instance
SETTINGS = Settings()


# =====================================
# CONVENIENCE FUNCTIONS
# =====================================

def get_settings() -> Settings:
    """Get global settings instance"""
    return SETTINGS


def update_detector_weights(weights_path: str):
    """
    Update detector weights path
    
    Args:
        weights_path: Path to YOLO weights file
    """
    SETTINGS.DETECTOR_WEIGHTS = weights_path
    print(f"✓ Updated detector weights: {weights_path}")


def enable_physics_mode():
    """Enable all physics features"""
    SETTINGS.PHYSICS_CONFIG['enable_eager'] = True
    SETTINGS.PHYSICS_CONFIG['enable_behaviour_detection'] = True
    print("✓ Physics mode enabled")


def disable_physics_mode():
    """Disable all physics features"""
    SETTINGS.PHYSICS_CONFIG['enable_eager'] = False
    SETTINGS.PHYSICS_CONFIG['enable_behaviour_detection'] = False
    SETTINGS.PHYSICS_CONFIG['enable_physics_predictor'] = False
    print("✓ Physics mode disabled")


if __name__ == "__main__":
    print("Testing Settings...")
    print(f"Device: {SETTINGS.DEVICE}")
    print(f"Detector config: {SETTINGS.get_detector_config()}")
    print(f"Tracking config: {SETTINGS.get_tracking_config()}")
    print("✓ Settings test completed")
