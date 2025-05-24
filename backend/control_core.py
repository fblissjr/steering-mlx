# control_core.py
"""
Multi-model control vector framework for MLX-LM models.
Extends the Gemma 3 POC to support multiple model architectures.
"""
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Tuple, List, Any, Union
import logging

logger = logging.getLogger(__name__)

# Universal control points that should be available across most transformer architectures
UNIVERSAL_CONTROL_POINTS = [
    "pre_attention_layernorm_input",
    "attention_output", 
    "post_attention_residual",
    "pre_mlp_layernorm_input", 
    "mlp_output",
    "post_mlp_residual",
]

# Model-specific control points that may not exist in all architectures
EXTENDED_CONTROL_POINTS = {
    "gemma3": [
        "query_norm_output",
        "key_norm_output", 
        "pre_feedforward_layernorm_input",
        "post_feedforward_layernorm_output",
    ],
    "mixtral": [
        "router_output",
        "expert_outputs",
        "expert_gate_weights",
    ],
    "llama": [
        # Llama uses standard points, no extensions needed
    ]
}

def get_control_points_for_model(model_type: str) -> List[str]:
    """Get available control points for a specific model type."""
    base_points = UNIVERSAL_CONTROL_POINTS.copy()
    extended_points = EXTENDED_CONTROL_POINTS.get(model_type.lower(), [])
    return base_points + extended_points

def clip_residual(x: mx.array, y: mx.array) -> mx.array:
    """
    Performs a clipped residual addition for float16 stability.
    Critical for models like Gemma 3 that use this pattern.
    """
    if x.dtype == mx.float16:
        bound = mx.finfo(mx.float16).max
        return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(mx.float16)
    return x + y

class ControlledLayerBase:
    """
    Base class for controlled layers across different model architectures.
    Provides common functionality for control vector application and activation capture.
    """
    
    def __init__(self, layer_idx: int, model_type: str):
        self.layer_idx = layer_idx
        self.model_type = model_type.lower()
        self.available_control_points = get_control_points_for_model(self.model_type)
        
        # Control vector storage: {control_point: [(vector, strength), ...]}
        self._control_vectors: Dict[str, List[Tuple[mx.array, float]]] = {
            cp: [] for cp in self.available_control_points
        }
        self._active_control_points: Dict[str, bool] = {
            cp: False for cp in self.available_control_points
        }
        
        # Activation capture storage
        self._capture_activations_map: Dict[str, Optional[mx.array]] = {
            cp: None for cp in self.available_control_points
        }
        self._is_capturing: bool = False
        self._capture_targets: List[str] = []

    def add_control(self, control_point: str, vector: mx.array, strength: float):
        """Add a control vector at the specified control point."""
        if control_point not in self.available_control_points:
            available = ", ".join(self.available_control_points)
            raise ValueError(f"Unknown control point '{control_point}' for {self.model_type}. Available: {available}")
        
        self._control_vectors[control_point].append((vector, strength))
        self._active_control_points[control_point] = True
        logger.info(f"Layer {self.layer_idx} ({self.model_type}): Added control to '{control_point}' with strength {strength:.2f}")

    def clear_controls(self, control_point: Optional[str] = None):
        """Clear control vectors at specified point or all points."""
        if control_point:
            if control_point in self.available_control_points:
                self._control_vectors[control_point] = []
                self._active_control_points[control_point] = False
                logger.info(f"Layer {self.layer_idx} ({self.model_type}): Cleared controls for '{control_point}'")
            else:
                raise ValueError(f"Unknown control point '{control_point}' for {self.model_type}")
        else:
            for cp in self.available_control_points:
                self._control_vectors[cp] = []
                self._active_control_points[cp] = False
            logger.info(f"Layer {self.layer_idx} ({self.model_type}): Cleared all controls")

    def _apply_controls_at_point(self, x: mx.array, point_name: str) -> mx.array:
        """Apply all active control vectors at the specified point."""
        if not self._active_control_points.get(point_name, False):
            return x
            
        for vec, strength in self._control_vectors[point_name]:
            if x.ndim == 3 and vec.ndim == 1:  # Broadcast 1D vector to 3D activation
                x = x + (vec * strength).astype(x.dtype)
            elif x.ndim == vec.ndim:  # Direct addition for matching dimensions
                x = x + (vec * strength).astype(x.dtype)
            else:
                logger.warning(
                    f"Layer {self.layer_idx}, Point '{point_name}': Vector shape {vec.shape} "
                    f"not broadcastable to activation shape {x.shape}. Skipping."
                )
        return x

    def _capture_if_needed(self, x: mx.array, point_name: str):
        """Capture activation if this point is being monitored."""
        if self._is_capturing and point_name in self._capture_targets:
            self._capture_activations_map[point_name] = x
            logger.debug(f"Layer {self.layer_idx} ({self.model_type}): Captured activation for '{point_name}' with shape {x.shape}")

    def start_capture(self, target_points: List[str]):
        """Start capturing activations at specified control points."""
        # Validate target points
        invalid_points = [p for p in target_points if p not in self.available_control_points]
        if invalid_points:
            available = ", ".join(self.available_control_points)
            raise ValueError(f"Invalid capture points for {self.model_type}: {invalid_points}. Available: {available}")
        
        self._is_capturing = True
        self._capture_targets = target_points
        for cp in self.available_control_points:
            self._capture_activations_map[cp] = None
        logger.info(f"Layer {self.layer_idx} ({self.model_type}): Started capturing for {target_points}")

    def stop_capture(self) -> Dict[str, Optional[mx.array]]:
        """Stop capturing and return captured activations."""
        self._is_capturing = False
        captured_data = {k: v for k, v in self._capture_activations_map.items() if v is not None}
        
        # Clear captured activations for the targeted points
        for cp in self._capture_targets:
            self._capture_activations_map[cp] = None
        self._capture_targets = []
        
        logger.info(f"Layer {self.layer_idx} ({self.model_type}): Stopped capturing. Captured: {list(captured_data.keys())}")
        return captured_data

# Model Architecture Registry
MODEL_ARCHITECTURES = {
    "gemma3": {
        "decoder_layer_class": "ControlledGemma3DecoderLayer",
        "model_class": "ControlledGemma3TextModel", 
        "shell_class": "mlx_lm.models.gemma3.Model",
        "requires_clip_residual": True,
        "sliding_window": True,
    },
    "llama": {
        "decoder_layer_class": "ControlledLlamaDecoderLayer",
        "model_class": "ControlledLlamaModel",
        "shell_class": "mlx_lm.models.llama.Model", 
        "requires_clip_residual": False,
        "sliding_window": False,
    },
    "mixtral": {
        "decoder_layer_class": "ControlledMixtralDecoderLayer", 
        "model_class": "ControlledMixtralModel",
        "shell_class": "mlx_lm.models.mixtral.Model",
        "requires_clip_residual": False,
        "sliding_window": False,
        "moe": True,
    }
}

def get_model_architecture_info(model_type: str) -> Dict[str, Any]:
    """Get architecture information for a model type."""
    model_type_lower = model_type.lower()
    if model_type_lower not in MODEL_ARCHITECTURES:
        available = ", ".join(MODEL_ARCHITECTURES.keys())
        raise ValueError(f"Unsupported model architecture: {model_type}. Available: {available}")
    return MODEL_ARCHITECTURES[model_type_lower]

def detect_model_type(model_config: Dict[str, Any]) -> str:
    """Detect model type from model configuration."""
    model_type = model_config.get("model_type", "").lower()
    
    # Handle special cases and aliases
    if model_type in ["gemma3_text", "gemma3text"]:
        return "gemma3"
    elif model_type in ["llama", "llama2", "code_llama"]:
        return "llama"
    elif model_type in ["mixtral"]:
        return "mixtral"
    
    # Direct mapping for known types
    if model_type in MODEL_ARCHITECTURES:
        return model_type
    
    raise ValueError(f"Unknown or unsupported model type: {model_type}")
