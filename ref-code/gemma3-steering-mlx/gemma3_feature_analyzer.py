# gemma3_feature_analyzer.py
import mlx.core as mx
import logging
from typing import List, Dict, Any, Callable, Optional

# Assuming gemma3_controlled_model.py and gemma3_control_core.py are in the same directory or PYTHONPATH
from gemma3_controlled_model import ControlledGemma3TextModel 
from gemma3_control_core import ControlledGemma3DecoderLayer, ALL_CONTROL_POINTS
from mlx_lm.models.gemma3 import Model as Gemma3ShellModel # For type hinting

logger = logging.getLogger(__name__)

def calculate_differentiation_metric(vec1: mx.array, vec2: mx.array, metric_type: str = "cosine") -> float:
    """Calculates a differentiation metric between two vectors."""
    if vec1 is None or vec2 is None:
        return float('-inf') # Or some other indicator of missing data
    if vec1.shape != vec2.shape:
        logger.warning(f"Vector shapes mismatch for metric calculation: {vec1.shape} vs {vec2.shape}")
        return float('-inf')
    
    if metric_type == "cosine_similarity":
        # Cosine similarity: (A . B) / (||A|| ||B||)
        # Higher is more similar. For differentiation, 1 - similarity or just similarity if context is clear.
        dot_product = mx.sum(vec1 * vec2)
        norm_vec1 = mx.linalg.norm(vec1)
        norm_vec2 = mx.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0 # Or handle as appropriate
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity.item()
    elif metric_type == "cosine_distance":
        # Cosine distance: 1 - cosine_similarity
        # Higher means more different.
        dot_product = mx.sum(vec1 * vec2)
        norm_vec1 = mx.linalg.norm(vec1)
        norm_vec2 = mx.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 1.0 
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return (1.0 - similarity).item()
    elif metric_type == "l2_distance":
        distance = mx.linalg.norm(vec1 - vec2)
        return distance.item()
    else:
        logger.warning(f"Unknown metric type: {metric_type}. Defaulting to L2 distance.")
        distance = mx.linalg.norm(vec1 - vec2)
        return distance.item()

def get_mean_activations_for_prompts(
    model_shell: Gemma3ShellModel,
    tokenizer: Any,
    prompts_raw: List[str],
    layer_idx: int,
    control_point: str,
    process_raw_prompt_func: Callable[[str], str],
    get_add_special_tokens_flag_func: Callable[[str], bool],
    average_over_tokens: bool = True
) -> Optional[mx.array]:
    """Helper to get mean activations for a list of prompts at a specific layer/point."""
    
    controlled_text_model: ControlledGemma3TextModel = model_shell.language_model
    if not (0 <= layer_idx < len(controlled_text_model.layers)):
        logger.error(f"Invalid layer_idx {layer_idx} for activation capture.")
        return None
    
    target_layer = controlled_text_model.layers[layer_idx]
    if not isinstance(target_layer, ControlledGemma3DecoderLayer):
        logger.error(f"Layer {layer_idx} is not a ControlledGemma3DecoderLayer.")
        return None

    all_activations_for_set = []
    target_layer.start_capture([control_point])

    for raw_prompt in prompts_raw:
        processed_prompt_str = process_raw_prompt_func(raw_prompt)
        add_special_tokens = get_add_special_tokens_flag_func(raw_prompt)
        input_ids = tokenizer.encode(processed_prompt_str, add_special_tokens=add_special_tokens)
        
        if not input_ids: # Handle empty tokenization
            logger.warning(f"Empty tokenization for prompt: '{raw_prompt[:50]}...'. Skipping for activation capture.")
            continue
        input_ids_mx = mx.array(input_ids)

        _ = model_shell(input_ids_mx) # Forward pass to trigger capture
        
        captured_data = target_layer.stop_capture() # stop_capture also clears targets for next call
        activation = captured_data.get(control_point)

        if activation is not None:
            # Activation shape is likely (batch_size=1, seq_len, hidden_dim)
            # or (seq_len, hidden_dim) if batch_size=1 was squeezed by model
            act_processed = activation
            if act_processed.ndim == 3 and act_processed.shape[0] == 1: # Squeeze batch if 1
                act_processed = act_processed.squeeze(0)
            
            if average_over_tokens and act_processed.ndim == 2: # (seq_len, hidden_dim)
                act_processed = mx.mean(act_processed, axis=0) # Now (hidden_dim,)
            elif not average_over_tokens and act_processed.ndim == 2: # Keep all token activations
                # For now, we'll still average if not averaging over tokens but multiple tokens exist,
                # to ensure a single vector per prompt for easy comparison.
                # A more advanced version might compare token-wise or use other pooling.
                logger.debug(f"average_over_tokens is False, but activation for prompt has shape {act_processed.shape}. Averaging for now.")
                act_processed = mx.mean(act_processed, axis=0)


            if act_processed.ndim == 1: # Ensure it's a 1D vector after processing
                 all_activations_for_set.append(act_processed)
            else:
                logger.warning(f"Activation for prompt '{raw_prompt[:50]}...' at L{layer_idx}|{control_point} has unexpected shape {activation.shape} after processing. Skipping.")
        else:
            logger.warning(f"No activation captured for point '{control_point}' in layer {layer_idx} for prompt: '{raw_prompt[:50]}...'")
    
    if not all_activations_for_set:
        return None
    
    return mx.mean(mx.stack(all_activations_for_set, axis=0), axis=0)


def analyze_feature_activation(
    model_shell: Gemma3ShellModel,
    tokenizer: Any,
    feature_positive_prompts_raw: List[str],
    feature_negative_prompts_raw: List[str],
    layers_to_analyze: List[int],
    control_points_to_analyze: List[str],
    process_raw_prompt_func: Callable[[str], str],
    get_add_special_tokens_flag_func: Callable[[str], bool],
    metric_type: str = "cosine_distance",
    average_activations_over_tokens: bool = True
):
    logger.info("Starting feature activation analysis...")
    results = []

    if not layers_to_analyze:
        num_model_layers = len(model_shell.language_model.layers)
        layers_to_analyze = list(range(num_model_layers)) # Default to all layers
        logger.info(f"No specific layers provided, analyzing all {num_model_layers} layers.")

    if not control_points_to_analyze:
        control_points_to_analyze = ALL_CONTROL_POINTS # Default to all control points
        logger.info(f"No specific control points provided, analyzing all {len(ALL_CONTROL_POINTS)} points.")

    for layer_idx in layers_to_analyze:
        for control_point in control_points_to_analyze:
            logger.info(f"Analyzing Layer {layer_idx}, Control Point: {control_point}")

            mean_pos_act = get_mean_activations_for_prompts(
                model_shell, tokenizer, feature_positive_prompts_raw, layer_idx, control_point,
                process_raw_prompt_func, get_add_special_tokens_flag_func, average_activations_over_tokens
            )
            mean_neg_act = get_mean_activations_for_prompts(
                model_shell, tokenizer, feature_negative_prompts_raw, layer_idx, control_point,
                process_raw_prompt_func, get_add_special_tokens_flag_func, average_activations_over_tokens
            )

            if mean_pos_act is not None and mean_neg_act is not None:
                metric_value = calculate_differentiation_metric(mean_pos_act, mean_neg_act, metric_type)
                results.append({
                    "layer_idx": layer_idx,
                    "control_point": control_point,
                    "metric_type": metric_type,
                    "differentiation_score": metric_value,
                    "norm_pos_vector": mx.linalg.norm(mean_pos_act).item() if mean_pos_act is not None else -1,
                    "norm_neg_vector": mx.linalg.norm(mean_neg_act).item() if mean_neg_act is not None else -1,
                })
                logger.debug(f"L{layer_idx}|{control_point}: {metric_type} = {metric_value:.4f}")
            else:
                logger.warning(f"Could not compute activations for L{layer_idx}|{control_point}. Skipping metric calculation.")
    
    # Sort results by differentiation score (higher is more differentiated for distance metrics)
    if metric_type in ["cosine_distance", "l2_distance"]:
        results.sort(key=lambda x: x["differentiation_score"], reverse=True)
    else: # For similarity, lower is more differentiated (or sort ascending)
        results.sort(key=lambda x: x["differentiation_score"])


    logger.info("\n--- Feature Activation Analysis Results ---")
    if results:
        for res in results:
            logger.info(f"Layer: {res['layer_idx']:<3} | Point: {res['control_point']:<30} | {res['metric_type']}: {res['differentiation_score']:.4f} (PosNorm: {res['norm_pos_vector']:.2f}, NegNorm: {res['norm_neg_vector']:.2f})")
    else:
        logger.info("No analysis results generated. Check logs for errors during activation capture.")
    logger.info("-----------------------------------------")
    return results
