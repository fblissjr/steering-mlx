# model_loader.py
"""
Unified model loader for multiple MLX-LM architectures.
Automatically detects model type and loads appropriate controlled implementation.
"""
import logging
from typing import Optional, Dict, Tuple, Any
from pathlib import Path

from mlx_lm.utils import get_model_path
from mlx_lm import load as load_mlx_model

from control_core import detect_model_type, get_model_architecture_info, MODEL_ARCHITECTURES
from controlled_gemma3 import load_controlled_gemma3_model
from controlled_llama import load_controlled_llama_model  
from controlled_mixtral import load_controlled_mixtral_model

logger = logging.getLogger(__name__)

class ModelLoadResult:
    """Result of model loading operation with metadata."""
    
    def __init__(
        self,
        model_shell: Any,
        tokenizer: Any,
        model_type: str,
        architecture_info: Dict[str, Any],
        model_path: str,
        is_controlled: bool = True
    ):
        self.model_shell = model_shell
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.architecture_info = architecture_info
        self.model_path = model_path
        self.is_controlled = is_controlled
        
        # Extract key metrics
        if hasattr(model_shell, 'language_model') and hasattr(model_shell.language_model, 'args'):
            args = model_shell.language_model.args
            self.num_layers = getattr(args, 'num_hidden_layers', 0)
            self.hidden_size = getattr(args, 'hidden_size', 0)
            self.vocab_size = getattr(args, 'vocab_size', 0)
        else:
            self.num_layers = 0
            self.hidden_size = 0
            self.vocab_size = 0

def load_controlled_model(
    model_name_or_path: str,
    tokenizer_config: Optional[Dict] = None,
    trust_remote_code: bool = True,
    force_model_type: Optional[str] = None
) -> ModelLoadResult:
    """
    Load a model with control capabilities, automatically detecting the architecture.
    
    Args:
        model_name_or_path: Path or HuggingFace model ID
        tokenizer_config: Optional tokenizer configuration
        trust_remote_code: Whether to trust remote code
        force_model_type: Force a specific model type (bypasses auto-detection)
        
    Returns:
        ModelLoadResult with loaded model and metadata
        
    Raises:
        ValueError: If model type is unsupported or loading fails
    """
    if tokenizer_config is None:
        tokenizer_config = {}

    logger.info(f"Loading model: {model_name_or_path}")
    
    # Resolve model path 
    try:
        model_path_resolved = get_model_path(model_name_or_path)
    except Exception as e:
        raise ValueError(f"Failed to resolve model path '{model_name_or_path}': {e}")

    # First, load the model normally to detect its type
    try:
        original_model, tokenizer = load_mlx_model(
            str(model_path_resolved), 
            tokenizer_config=tokenizer_config
        )
    except Exception as e:
        raise ValueError(f"Failed to load model '{model_name_or_path}': {e}")

    # Detect model type
    if force_model_type:
        model_type = force_model_type.lower()
        logger.info(f"Using forced model type: {model_type}")
    else:
        try:
            # Try to get model type from config
            config_args = getattr(original_model, 'args', None)
            if config_args:
                if hasattr(config_args, 'text_config'):
                    config_dict = config_args.text_config
                else:
                    config_dict = vars(config_args)
                model_type = detect_model_type(config_dict)
            else:
                raise ValueError("Cannot extract model configuration")
        except Exception as e:
            logger.error(f"Failed to detect model type: {e}")
            raise ValueError(f"Unsupported or undetectable model type for '{model_name_or_path}'")

    logger.info(f"Detected model type: {model_type}")

    # Load controlled version based on model type
    controlled_model = None
    controlled_tokenizer = None
    
    if model_type == "gemma3":
        try:
            controlled_model, controlled_tokenizer = load_controlled_gemma3_model(
                model_name_or_path, tokenizer_config, trust_remote_code
            )
        except Exception as e:
            logger.error(f"Failed to load controlled Gemma 3 model: {e}")
            raise ValueError(f"Failed to load controlled Gemma 3 model: {e}")
            
    elif model_type == "llama":
        # For now, return placeholder - Llama implementation is not complete
        logger.warning("Llama controlled model is not fully implemented yet")
        controlled_model, controlled_tokenizer = original_model, tokenizer
        
    elif model_type == "mixtral":
        # For now, return placeholder - Mixtral implementation is not complete  
        logger.warning("Mixtral controlled model is not fully implemented yet")
        controlled_model, controlled_tokenizer = original_model, tokenizer
        
    else:
        available_types = ", ".join(MODEL_ARCHITECTURES.keys())
        raise ValueError(f"Unsupported model type '{model_type}'. Available: {available_types}")

    # Get architecture information
    try:
        if hasattr(controlled_model, 'language_model') and hasattr(controlled_model.language_model, 'get_architecture_info'):
            architecture_info = controlled_model.language_model.get_architecture_info()
        else:
            # Fallback architecture info
            architecture_info = get_basic_architecture_info(controlled_model, model_type)
    except Exception as e:
        logger.warning(f"Failed to get detailed architecture info: {e}")
        architecture_info = get_basic_architecture_info(controlled_model, model_type)

    # Create result
    result = ModelLoadResult(
        model_shell=controlled_model,
        tokenizer=controlled_tokenizer,
        model_type=model_type,
        architecture_info=architecture_info,
        model_path=str(model_name_or_path),
        is_controlled=(model_type == "gemma3")  # Only Gemma 3 is fully controlled for now
    )
    
    logger.info(f"Successfully loaded {model_type} model: {result.num_layers} layers, "
               f"hidden_size={result.hidden_size}, vocab_size={result.vocab_size}")
    
    return result

def get_basic_architecture_info(model_shell: Any, model_type: str) -> Dict[str, Any]:
    """
    Extract basic architecture info when detailed info is not available.
    """
    try:
        if hasattr(model_shell, 'language_model') and hasattr(model_shell.language_model, 'args'):
            args = model_shell.language_model.args
            num_layers = getattr(args, 'num_hidden_layers', 0)
            hidden_size = getattr(args, 'hidden_size', 0)
            vocab_size = getattr(args, 'vocab_size', 0)
        elif hasattr(model_shell, 'args'):
            args = model_shell.args
            num_layers = getattr(args, 'num_hidden_layers', 0)
            hidden_size = getattr(args, 'hidden_size', 0) 
            vocab_size = getattr(args, 'vocab_size', 0)
        else:
            # Fallback defaults
            num_layers = 32
            hidden_size = 4096
            vocab_size = 32000

        return {
            "model_name": f"Controlled{model_type.title()}Model",
            "model_type": model_type,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "sliding_window": model_type == "gemma3",
            "moe": model_type == "mixtral",
            "layer_structure": {
                "embed_tokens": {
                    "type": "Embedding",
                    "params": {"vocab_size": vocab_size, "hidden_size": hidden_size}
                },
                "layers": {
                    "type": "DecoderLayerList",
                    "count": num_layers,
                    "item_structure": {
                        "type": f"Controlled{model_type.title()}DecoderLayer",
                        "control_points": []  # Will be filled by specific implementations
                    }
                },
                "norm": {"type": "RMSNorm"},
                "lm_head": {"type": "Linear", "params": {"out_features": vocab_size}}
            }
        }
    except Exception as e:
        logger.error(f"Failed to extract basic architecture info: {e}")
        return {
            "model_name": f"Unknown{model_type.title()}Model",
            "model_type": model_type,
            "num_layers": 0,
            "hidden_size": 0,
            "vocab_size": 0,
            "error": str(e)
        }

def get_supported_model_types() -> List[str]:
    """Get list of supported model types."""
    return list(MODEL_ARCHITECTURES.keys())

def is_model_type_supported(model_type: str) -> bool:
    """Check if a model type is supported."""
    return model_type.lower() in MODEL_ARCHITECTURES

def get_model_type_info(model_type: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific model type."""
    try:
        return get_model_architecture_info(model_type)
    except ValueError:
        return None
