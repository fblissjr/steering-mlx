# control_utils.py
"""
Multi-model control vector utilities.
Handles control vector derivation, loading, and management across different architectures.
"""
import mlx.core as mx
import numpy as np
import os
from typing import Optional, List, Any, Dict, Callable, Union
import logging

from control_core import ControlledLayerBase, detect_model_type, get_control_points_for_model

logger = logging.getLogger(__name__)

def derive_control_vector(
    model_shell: Any,
    tokenizer: Any,
    positive_prompts: List[str],
    negative_prompts: List[str],
    layer_idx: int,
    control_point: str,
    average_over_tokens: bool = True,
    model_type: Optional[str] = None
) -> Optional[mx.array]:
    """
    Derive a control vector from positive and negative prompt sets.
    Works across different model architectures.
    """
    # Auto-detect model type if not provided
    if model_type is None:
        try:
            if hasattr(model_shell, 'language_model') and hasattr(model_shell.language_model, 'model_type'):
                model_type = model_shell.language_model.model_type
            else:
                # Try to detect from config
                config_args = getattr(model_shell, 'args', None)
                if config_args:
                    config_dict = getattr(config_args, 'text_config', vars(config_args))
                    model_type = detect_model_type(config_dict)
                else:
                    raise ValueError("Cannot determine model type")
        except Exception as e:
            logger.error(f"Failed to detect model type: {e}")
            return None

    controlled_text_model = model_shell.language_model
    
    # Validate control point for this model type
    available_points = get_control_points_for_model(model_type)
    if control_point not in available_points:
        logger.error(f"Control point '{control_point}' not available for model type '{model_type}'. Available: {available_points}")
        return None

    def get_activations_for_prompts(prompts: List[str]) -> List[mx.array]:
        """Get activations for a set of prompts."""
        all_activations = []
        
        if not (0 <= layer_idx < len(controlled_text_model.layers)):
            logger.error(f"Invalid layer_idx {layer_idx} for model with {len(controlled_text_model.layers)} layers")
            return []
        
        target_layer = controlled_text_model.layers[layer_idx]
        if not isinstance(target_layer, ControlledLayerBase):
            logger.error(f"Layer {layer_idx} is not a ControlledLayerBase instance")
            return []

        target_layer.start_capture([control_point])
        
        for prompt_text in prompts:
            try:
                # Handle different tokenizer formats
                if hasattr(tokenizer, 'encode'):
                    input_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
                else:
                    # Fallback for different tokenizer types
                    input_ids = tokenizer(prompt_text, return_tensors="np")["input_ids"][0]
                
                if len(input_ids) == 0:
                    logger.warning(f"Empty tokenization for prompt: '{prompt_text}'. Skipping.")
                    continue

                input_ids_mx = mx.array([input_ids])  # Add batch dimension
                
                # Forward pass to capture activations
                _ = model_shell(input_ids_mx)
                
                captured_data = target_layer.stop_capture()
                activation = captured_data.get(control_point)
                
                if activation is not None:
                    # Process activation based on averaging preference
                    processed_activation = process_activation(activation, average_over_tokens)
                    if processed_activation is not None:
                        all_activations.append(processed_activation)
                else:
                    logger.warning(f"No activation captured for point '{control_point}' in layer {layer_idx} for prompt: '{prompt_text[:50]}...'")
                    
            except Exception as e:
                logger.error(f"Error processing prompt '{prompt_text[:50]}...': {e}")
                continue
        
        return all_activations

    logger.info(f"Deriving control vector for {model_type} model, layer {layer_idx}, point '{control_point}'...")
    pos_activations = get_activations_for_prompts(positive_prompts)
    neg_activations = get_activations_for_prompts(negative_prompts)

    if not pos_activations or not neg_activations:
        logger.warning("Could not retrieve sufficient activations for deriving control vector.")
        return None

    try:
        # Stack and average activations
        mean_pos_act = mx.mean(mx.stack(pos_activations, axis=0), axis=0)
        mean_neg_act = mx.mean(mx.stack(neg_activations, axis=0), axis=0)
        
        # Derive control vector as difference
        control_vec = mean_pos_act - mean_neg_act
        
        logger.info(f"Derived control vector for {model_type} layer {layer_idx}, point '{control_point}'. "
                   f"Shape: {control_vec.shape}, Norm: {mx.linalg.norm(control_vec):.4f}")
        return control_vec
        
    except Exception as e:
        logger.error(f"Error computing control vector: {e}")
        logger.error(f"Positive activation shapes: {[a.shape for a in pos_activations]}")
        logger.error(f"Negative activation shapes: {[a.shape for a in neg_activations]}")
        return None

def process_activation(activation: mx.array, average_over_tokens: bool = True) -> Optional[mx.array]:
    """
    Process captured activation into a consistent format.
    """
    if activation is None:
        return None

    # Handle different activation shapes
    if activation.ndim == 3 and activation.shape[0] == 1:  # (1, seq_len, hidden_dim)
        activation = activation.squeeze(0)  # (seq_len, hidden_dim)
    
    if average_over_tokens and activation.ndim == 2:  # (seq_len, hidden_dim)
        return mx.mean(activation, axis=0)  # (hidden_dim,)
    elif activation.ndim == 1:  # Already (hidden_dim,)
        return activation
    elif not average_over_tokens and activation.ndim == 2:
        # For now, still average to get consistent shapes
        # Future versions could support token-level analysis
        return mx.mean(activation, axis=0)
    else:
        logger.warning(f"Unexpected activation shape: {activation.shape}")
        return None

def load_control_vector(file_path: str, expected_shape: Optional[tuple] = None) -> Optional[mx.array]:
    """
    Load a control vector from file.
    Supports .npy and .npz formats.
    """
    if not os.path.exists(file_path):
        logger.error(f"Control vector file not found: {file_path}")
        return None

    try:
        # Load using MLX's native loading
        loaded_data = mx.load(file_path)
        
        control_vector = None
        if isinstance(loaded_data, dict):
            # Handle .npz files
            if 'vector' in loaded_data:
                control_vector = loaded_data['vector']
            elif 'arr_0' in loaded_data:
                control_vector = loaded_data['arr_0']
            elif len(loaded_data.keys()) == 1:
                control_vector = list(loaded_data.values())[0]
            else:
                logger.error(f"NPZ file {file_path} has multiple arrays and no 'vector' or 'arr_0' key")
                return None
        elif isinstance(loaded_data, mx.array):
            control_vector = loaded_data
        else:
            logger.error(f"Loaded data from {file_path} is not an mx.array or recognized dict format")
            return None
        
        # Validate shape if expected shape is provided
        if expected_shape and control_vector.shape != expected_shape:
            logger.error(f"Loaded vector from {file_path} has shape {control_vector.shape}, expected {expected_shape}")
            return None
        
        logger.info(f"Loaded control vector from {file_path} with shape {control_vector.shape}")
        return control_vector
        
    except Exception as e:
        logger.error(f"Failed to load control vector from {file_path}: {e}")
        return None

def save_control_vector(vector: mx.array, file_path: str, metadata: Optional[Dict] = None) -> bool:
    """
    Save a control vector to file with optional metadata.
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if metadata:
            # Save as .npz with metadata
            save_dict = {"vector": vector}
            save_dict.update(metadata)
            mx.savez(file_path, **save_dict)
        else:
            # Save as simple .npy
            mx.save(file_path, vector)
        
        logger.info(f"Saved control vector to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save control vector to {file_path}: {e}")
        return False

def create_random_control_vector(
    shape: tuple, 
    vector_type: str = "positive", 
    scale: float = 0.1
) -> mx.array:
    """
    Create a random control vector for testing purposes.
    """
    base_vector = mx.random.normal(shape=shape).astype(mx.float16)
    multiplier = 1.0 if vector_type == "positive" else -1.0
    return base_vector * scale * multiplier

def get_model_hidden_size(model_shell: Any) -> int:
    """
    Extract hidden size from a model shell.
    """
    try:
        if hasattr(model_shell, 'language_model') and hasattr(model_shell.language_model, 'args'):
            return model_shell.language_model.args.hidden_size
        elif hasattr(model_shell, 'args'):
            args = model_shell.args
            if hasattr(args, 'text_config'):
                return args.text_config.get('hidden_size', args.hidden_size)
            return args.hidden_size
        else:
            raise AttributeError("Cannot find hidden_size attribute")
    except Exception as e:
        logger.error(f"Failed to get model hidden size: {e}")
        return 4096  # Default fallback

def clear_all_controls(model_shell: Any):
    """
    Clear all active controls from all layers of a model.
    """
    controlled_text_model = model_shell.language_model
    
    cleared_count = 0
    for layer in controlled_text_model.layers:
        if isinstance(layer, ControlledLayerBase):
            layer.clear_controls()
            cleared_count += 1
    
    logger.info(f"Cleared all controls from {cleared_count} layers")
