# feature_analyzer.py
"""
Multi-model feature analysis utilities.
Analyzes where specific features are encoded within different model architectures.
"""
import mlx.core as mx
import logging
from typing import List, Dict, Any, Callable, Optional, Tuple

from control_core import ControlledLayerBase, detect_model_type, get_control_points_for_model
from control_utils import process_activation

logger = logging.getLogger(__name__)

def calculate_differentiation_metric(vec1: mx.array, vec2: mx.array, metric_type: str = "cosine_distance") -> float:
    """
    Calculate differentiation metric between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector  
        metric_type: Type of metric ("cosine_distance", "cosine_similarity", "l2_distance")
    
    Returns:
        Differentiation score as float
    """
    if vec1 is None or vec2 is None:
        return float('-inf')
    if vec1.shape != vec2.shape:
        logger.warning(f"Vector shapes mismatch: {vec1.shape} vs {vec2.shape}")
        return float('-inf')
    
    try:
        if metric_type == "cosine_similarity":
            dot_product = mx.sum(vec1 * vec2)
            norm_vec1 = mx.linalg.norm(vec1)
            norm_vec2 = mx.linalg.norm(vec2)
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0.0
            similarity = dot_product / (norm_vec1 * norm_vec2)
            return similarity.item()
            
        elif metric_type == "cosine_distance":
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
            logger.warning(f"Unknown metric type: {metric_type}. Using L2 distance.")
            distance = mx.linalg.norm(vec1 - vec2)
            return distance.item()
            
    except Exception as e:
        logger.error(f"Error calculating {metric_type}: {e}")
        return float('-inf')

def get_mean_activations_for_prompts(
    model_shell: Any,
    tokenizer: Any, 
    prompts_raw: List[str],
    layer_idx: int,
    control_point: str,
    process_raw_prompt_func: Callable[[str], str],
    get_add_special_tokens_flag_func: Callable[[str], bool],
    average_over_tokens: bool = True,
    model_type: Optional[str] = None
) -> Optional[mx.array]:
    """
    Get mean activations for a list of prompts at a specific layer/control point.
    Works across different model architectures.
    """
    # Auto-detect model type if not provided
    if model_type is None:
        try:
            if hasattr(model_shell, 'language_model') and hasattr(model_shell.language_model, 'model_type'):
                model_type = model_shell.language_model.model_type
            else:
                config_args = getattr(model_shell, 'args', None)
                if config_args:
                    config_dict = getattr(config_args, 'text_config', vars(config_args))
                    model_type = detect_model_type(config_dict)
                else:
                    model_type = "unknown"
        except Exception as e:
            logger.warning(f"Could not detect model type: {e}")
            model_type = "unknown"

    controlled_text_model = model_shell.language_model
    
    if not (0 <= layer_idx < len(controlled_text_model.layers)):
        logger.error(f"Invalid layer_idx {layer_idx} for model with {len(controlled_text_model.layers)} layers")
        return None
    
    target_layer = controlled_text_model.layers[layer_idx]
    if not isinstance(target_layer, ControlledLayerBase):
        logger.error(f"Layer {layer_idx} is not a ControlledLayerBase instance")
        return None

    # Validate control point for this model type
    available_points = get_control_points_for_model(model_type)
    if model_type != "unknown" and control_point not in available_points:
        logger.warning(f"Control point '{control_point}' may not be available for model type '{model_type}'")

    all_activations_for_set = []
    target_layer.start_capture([control_point])

    for raw_prompt in prompts_raw:
        try:
            processed_prompt_str = process_raw_prompt_func(raw_prompt)
            add_special_tokens = get_add_special_tokens_flag_func(raw_prompt)
            
            # Handle different tokenizer formats
            if hasattr(tokenizer, 'encode'):
                input_ids = tokenizer.encode(processed_prompt_str, add_special_tokens=add_special_tokens)
            else:
                # Fallback for different tokenizer implementations
                tokenized = tokenizer(processed_prompt_str, add_special_tokens=add_special_tokens, return_tensors="np")
                input_ids = tokenized["input_ids"][0] if "input_ids" in tokenized else tokenized
            
            if not input_ids:
                logger.warning(f"Empty tokenization for prompt: '{raw_prompt[:50]}...'. Skipping.")
                continue
                
            input_ids_mx = mx.array([input_ids])  # Add batch dimension

            # Forward pass to trigger activation capture
            _ = model_shell(input_ids_mx)
            
            captured_data = target_layer.stop_capture()
            activation = captured_data.get(control_point)

            if activation is not None:
                processed_activation = process_activation(activation, average_over_tokens)
                if processed_activation is not None and processed_activation.ndim == 1:
                    all_activations_for_set.append(processed_activation)
                else:
                    logger.warning(f"Invalid processed activation shape for prompt '{raw_prompt[:50]}...'")
            else:
                logger.warning(f"No activation captured for point '{control_point}' in layer {layer_idx}")
                
        except Exception as e:
            logger.error(f"Error processing prompt '{raw_prompt[:50]}...': {e}")
            continue
    
    if not all_activations_for_set:
        return None
    
    try:
        return mx.mean(mx.stack(all_activations_for_set, axis=0), axis=0)
    except Exception as e:
        logger.error(f"Error computing mean activations: {e}")
        return None

def analyze_feature_activation(
    model_shell: Any,
    tokenizer: Any,
    feature_positive_prompts_raw: List[str],
    feature_negative_prompts_raw: List[str], 
    layers_to_analyze: List[int],
    control_points_to_analyze: List[str],
    process_raw_prompt_func: Callable[[str], str],
    get_add_special_tokens_flag_func: Callable[[str], bool],
    metric_type: str = "cosine_distance",
    average_activations_over_tokens: bool = True,
    model_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Analyze feature activation patterns across layers and control points.
    Works with different model architectures.
    """
    logger.info("Starting multi-model feature activation analysis...")
    
    # Auto-detect model type if not provided
    if model_type is None:
        try:
            if hasattr(model_shell, 'language_model') and hasattr(model_shell.language_model, 'model_type'):
                model_type = model_shell.language_model.model_type
            else:
                config_args = getattr(model_shell, 'args', None)
                if config_args:
                    config_dict = getattr(config_args, 'text_config', vars(config_args))
                    model_type = detect_model_type(config_dict)
                else:
                    model_type = "unknown"
        except Exception as e:
            logger.warning(f"Could not detect model type: {e}")
            model_type = "unknown"

    results = []
    controlled_text_model = model_shell.language_model

    # Set defaults if not specified
    if not layers_to_analyze:
        num_model_layers = len(controlled_text_model.layers)
        layers_to_analyze = list(range(num_model_layers))
        logger.info(f"No specific layers provided, analyzing all {num_model_layers} layers")

    if not control_points_to_analyze:
        control_points_to_analyze = get_control_points_for_model(model_type)
        logger.info(f"No specific control points provided, analyzing all available points for {model_type}")

    total_combinations = len(layers_to_analyze) * len(control_points_to_analyze)
    logger.info(f"Analyzing {total_combinations} layer/control-point combinations")

    for layer_idx in layers_to_analyze:
        for control_point in control_points_to_analyze:
            logger.info(f"Analyzing {model_type} Layer {layer_idx}, Control Point: {control_point}")

            mean_pos_act = get_mean_activations_for_prompts(
                model_shell, tokenizer, feature_positive_prompts_raw, layer_idx, control_point,
                process_raw_prompt_func, get_add_special_tokens_flag_func, 
                average_activations_over_tokens, model_type
            )
            
            mean_neg_act = get_mean_activations_for_prompts(
                model_shell, tokenizer, feature_negative_prompts_raw, layer_idx, control_point,
                process_raw_prompt_func, get_add_special_tokens_flag_func,
                average_activations_over_tokens, model_type
            )

            if mean_pos_act is not None and mean_neg_act is not None:
                metric_value = calculate_differentiation_metric(mean_pos_act, mean_neg_act, metric_type)
                
                result = {
                    "model_type": model_type,
                    "layer_idx": layer_idx,
                    "control_point": control_point,
                    "metric_type": metric_type,
                    "differentiation_score": metric_value,
                    "norm_pos_vector": mx.linalg.norm(mean_pos_act).item() if mean_pos_act is not None else -1,
                    "norm_neg_vector": mx.linalg.norm(mean_neg_act).item() if mean_neg_act is not None else -1,
                }
                results.append(result)
                
                logger.debug(f"{model_type} L{layer_idx}|{control_point}: {metric_type} = {metric_value:.4f}")
            else:
                logger.warning(f"Could not compute activations for {model_type} L{layer_idx}|{control_point}")

    # Sort results by differentiation score
    if metric_type in ["cosine_distance", "l2_distance"]:
        results.sort(key=lambda x: x["differentiation_score"], reverse=True)
    else:  # For similarity metrics
        results.sort(key=lambda x: x["differentiation_score"])

    # Log results summary
    logger.info(f"\n--- {model_type.upper()} Feature Activation Analysis Results ---")
    if results:
        logger.info(f"Top 10 most differentiated locations:")
        for i, res in enumerate(results[:10]):
            logger.info(f"{i+1:2d}. Layer: {res['layer_idx']:<3} | Point: {res['control_point']:<30} | "
                       f"{res['metric_type']}: {res['differentiation_score']:.4f} "
                       f"(PosNorm: {res['norm_pos_vector']:.2f}, NegNorm: {res['norm_neg_vector']:.2f})")
    else:
        logger.info("No analysis results generated. Check logs for errors.")
    logger.info("-" * 60)
    
    return results

def compare_feature_across_models(
    model_results: Dict[str, List[Dict[str, Any]]],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Compare feature analysis results across different model architectures.
    
    Args:
        model_results: Dict mapping model_type to analysis results 
        top_k: Number of top results to compare per model
        
    Returns:
        Comparison summary
    """
    comparison = {
        "model_types": list(model_results.keys()),
        "top_locations_per_model": {},
        "common_patterns": [],
        "unique_patterns": {}
    }
    
    # Extract top locations for each model
    for model_type, results in model_results.items():
        if results:
            top_results = results[:top_k]
            comparison["top_locations_per_model"][model_type] = [
                f"L{r['layer_idx']}|{r['control_point']}" for r in top_results
            ]
    
    # Find common patterns (same layer/control point combinations that appear in top results)
    all_locations = set()
    for model_type, results in model_results.items():
        if results:
            for r in results[:top_k]:
                all_locations.add(f"L{r['layer_idx']}|{r['control_point']}")
    
    for location in all_locations:
        models_with_location = []
        for model_type, top_locations in comparison["top_locations_per_model"].items():
            if location in top_locations:
                models_with_location.append(model_type)
        
        if len(models_with_location) > 1:
            comparison["common_patterns"].append({
                "location": location,
                "models": models_with_location
            })
    
    logger.info(f"Feature comparison across {len(model_results)} model types completed")
    return comparison
