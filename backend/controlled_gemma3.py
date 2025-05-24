# controlled_gemma3.py
"""
Gemma 3 specific controlled model implementation.
Adapted from the CLI POC with improvements for the web UI.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gemma3 import Model as Gemma3ShellModel, ModelArgs as Gemma3ShellConfig
from mlx_lm.models.gemma3_text import ModelArgs as Gemma3TextConfig, TransformerBlock as OriginalGemma3DecoderLayer
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.utils import get_model_path
from mlx_lm import load as load_mlx_model
from mlx.utils import tree_flatten, tree_unflatten
from typing import Optional, Dict, Tuple, Any
import logging

from control_core import ControlledLayerBase, clip_residual, get_control_points_for_model

logger = logging.getLogger(__name__)

class ControlledGemma3DecoderLayer(OriginalGemma3DecoderLayer, ControlledLayerBase):
    """
    Gemma 3 decoder layer with control vector support.
    Extends the original TransformerBlock with intervention capabilities.
    """
    
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        OriginalGemma3DecoderLayer.__init__(self, config, layer_idx)
        ControlledLayerBase.__init__(self, layer_idx, "gemma3")

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        
        # 1. Input Layernorm for Attention
        x_ln1_input = x
        self._capture_if_needed(x_ln1_input, "pre_attention_layernorm_input")
        x_ln1_input_controlled = self._apply_controls_at_point(x_ln1_input, "pre_attention_layernorm_input")
        normed_x_for_attn = self.input_layernorm(x_ln1_input_controlled)

        # 2. Self-Attention
        attn_out_raw = self.self_attn(normed_x_for_attn, mask=mask, cache=cache)
        self._capture_if_needed(attn_out_raw, "attention_output")
        attn_out_controlled = self._apply_controls_at_point(attn_out_raw, "attention_output")
        
        # 3. First Residual Connection
        post_attn_ln_out = self.post_attention_layernorm(attn_out_controlled)
        h_after_attn = clip_residual(x, post_attn_ln_out)
        
        self._capture_if_needed(h_after_attn, "post_attention_residual")
        h_after_attn_controlled = self._apply_controls_at_point(h_after_attn, "post_attention_residual")

        # 4. Pre-Feedforward Layernorm
        mlp_ln_input = h_after_attn_controlled
        self._capture_if_needed(mlp_ln_input, "pre_mlp_layernorm_input")
        mlp_ln_input_controlled = self._apply_controls_at_point(mlp_ln_input, "pre_mlp_layernorm_input")
        normed_h_for_mlp = self.pre_feedforward_layernorm(mlp_ln_input_controlled)

        # 5. MLP
        mlp_out_raw = self.mlp(normed_h_for_mlp)
        self._capture_if_needed(mlp_out_raw, "mlp_output")
        mlp_out_controlled = self._apply_controls_at_point(mlp_out_raw, "mlp_output")

        # 6. Second Residual Connection
        post_ffw_ln_out = self.post_feedforward_layernorm(mlp_out_controlled)
        final_out = clip_residual(h_after_attn_controlled, post_ffw_ln_out)

        self._capture_if_needed(final_out, "post_mlp_residual")
        final_out_controlled = self._apply_controls_at_point(final_out, "post_mlp_residual")
        
        return final_out_controlled

class ControlledGemma3TextModel(nn.Module):
    """
    Controlled version of Gemma 3 text model with intervention capabilities.
    """
    
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            ControlledGemma3DecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        
        h *= mx.array(self.args.hidden_size**0.5, dtype=h.dtype)

        if cache is None:
            cache = self.make_cache()

        prepared_full_mask = None
        prepared_sliding_window_mask = None

        if mask is None and h.shape[1] > 1:
            from mlx_lm.models.base import create_attention_mask

            global_layer_idx_for_mask = self.args.sliding_window_pattern - 1
            if global_layer_idx_for_mask < 0:
                global_layer_idx_for_mask = 0

            cache_for_full_mask_list = None
            if cache and len(cache) > global_layer_idx_for_mask and cache[global_layer_idx_for_mask] is not None:
                cache_for_full_mask_list = [cache[global_layer_idx_for_mask]]
            
            prepared_full_mask = create_attention_mask(h, cache_for_full_mask_list)
            if isinstance(prepared_full_mask, str) and prepared_full_mask == "causal":
                prepared_full_mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            if prepared_full_mask is not None:
                prepared_full_mask = prepared_full_mask.astype(h.dtype)

            cache_for_sliding_mask_list = [cache[0]] if cache and len(cache) > 0 and cache[0] is not None else None
            prepared_sliding_window_mask = create_attention_mask(h, cache_for_sliding_mask_list)
            if isinstance(prepared_sliding_window_mask, str) and prepared_sliding_window_mask == "causal":
                prepared_sliding_window_mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            if prepared_sliding_window_mask is not None:
                prepared_sliding_window_mask = prepared_sliding_window_mask.astype(h.dtype)
        
        for i, layer_instance in enumerate(self.layers):
            layer_cache_obj = cache[i] if cache and i < len(cache) else None
            
            current_layer_mask = mask
            if current_layer_mask is None:
                is_global_layer = (i % self.args.sliding_window_pattern == self.args.sliding_window_pattern - 1)
                if is_global_layer:
                    current_layer_mask = prepared_full_mask
                else:
                    current_layer_mask = prepared_sliding_window_mask
            
            h = layer_instance(h, mask=current_layer_mask, cache=layer_cache_obj)
        
        final_norm_output = self.norm(h)
        out_logits = self.lm_head(final_norm_output)
        return out_logits

    @property
    def layers_prop(self):
        return self.layers

    def make_cache(self):
        caches = []
        for i in range(self.args.num_hidden_layers):
            if (i % self.args.sliding_window_pattern == self.args.sliding_window_pattern - 1):
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window, keep=0))
        return caches

    def sanitize(self, weights: Dict[str, Any]):
        """
        Sanitize weights from original model structure to controlled model structure.
        Handles key remapping and weight tying.
        """
        sanitized_weights = {}
        for k, v in weights.items():
            if k.startswith("model."):
                sanitized_weights[k[len("model."):]] = v
            elif k == "lm_head.weight":
                sanitized_weights[k] = v
            else:
                logger.warning(f"Unexpected weight key during sanitize: {k}. Using as is.")
                sanitized_weights[k] = v
        
        # Tie lm_head to embed_tokens if lm_head is missing
        if "lm_head.weight" not in sanitized_weights and "embed_tokens.weight" in sanitized_weights:
            logger.info("Tying lm_head.weight to embed_tokens.weight in ControlledGemma3TextModel.")
            sanitized_weights["lm_head.weight"] = sanitized_weights["embed_tokens.weight"]
        
        return sanitized_weights

    def get_architecture_info(self) -> Dict[str, Any]:
        """Return detailed architecture information for UI visualization."""
        return {
            "model_name": "ControlledGemma3TextModel",
            "model_type": "gemma3",
            "num_layers": self.args.num_hidden_layers,
            "hidden_size": self.args.hidden_size,
            "vocab_size": self.args.vocab_size,
            "sliding_window": True,
            "sliding_window_size": getattr(self.args, 'sliding_window', None),
            "sliding_window_pattern": getattr(self.args, 'sliding_window_pattern', None),
            "layer_structure": {
                "embed_tokens": {
                    "type": "Embedding",
                    "params": {"vocab_size": self.args.vocab_size, "hidden_size": self.args.hidden_size}
                },
                "layers": {
                    "type": "DecoderLayerList",
                    "count": self.args.num_hidden_layers,
                    "item_structure": {
                        "type": "ControlledGemma3DecoderLayer",
                        "sub_modules": [
                            {"name": "input_layernorm", "type": "RMSNorm"},
                            {"name": "self_attn", "type": "Attention", "sub_modules": [
                                {"name": "q_proj", "type": "Linear"},
                                {"name": "k_proj", "type": "Linear"},
                                {"name": "v_proj", "type": "Linear"},
                                {"name": "o_proj", "type": "Linear"},
                                {"name": "q_norm", "type": "RMSNorm"},
                                {"name": "k_norm", "type": "RMSNorm"},
                            ]},
                            {"name": "post_attention_layernorm", "type": "RMSNorm"},
                            {"name": "pre_feedforward_layernorm", "type": "RMSNorm"},
                            {"name": "mlp", "type": "MLP", "sub_modules": [
                                {"name": "gate_proj", "type": "Linear"},
                                {"name": "up_proj", "type": "Linear"},
                                {"name": "down_proj", "type": "Linear"},
                            ]},
                            {"name": "post_feedforward_layernorm", "type": "RMSNorm"},
                        ],
                        "control_points": get_control_points_for_model("gemma3")
                    }
                },
                "norm": {"type": "RMSNorm"},
                "lm_head": {"type": "Linear", "params": {"out_features": self.args.vocab_size}}
            }
        }

def load_controlled_gemma3_model(
    model_name_or_path: str, 
    tokenizer_config: Optional[Dict] = None,
    trust_remote_code: bool = True
) -> Tuple[Any, Any]:
    """
    Load a Gemma 3 model and wrap it with control capabilities.
    """
    if tokenizer_config is None:
        tokenizer_config = {}
        
    model_path_resolved = get_model_path(model_name_or_path)
    original_shell_model_instance, tokenizer = load_mlx_model(str(model_path_resolved), tokenizer_config=tokenizer_config)

    if not isinstance(original_shell_model_instance, Gemma3ShellModel):
        raise TypeError(f"Loaded model from {model_name_or_path} is not a Gemma3ShellModel. Type: {type(original_shell_model_instance)}")

    shell_config: Gemma3ShellConfig = original_shell_model_instance.args
    text_model_config_dict = shell_config.text_config
    
    if "vocab_size" not in text_model_config_dict or text_model_config_dict["vocab_size"] != shell_config.vocab_size:
        text_model_config_dict["vocab_size"] = shell_config.vocab_size

    text_model_args = Gemma3TextConfig.from_dict(text_model_config_dict)
    controlled_text_model = ControlledGemma3TextModel(text_model_args)
    
    original_language_model = original_shell_model_instance.language_model
    original_language_model_params_flat = dict(tree_flatten(original_language_model.parameters()))
    
    # Sanitize parameter keys before updating the controlled_text_model
    sanitized_params_flat = controlled_text_model.sanitize(original_language_model_params_flat)
    
    controlled_text_model.update(tree_unflatten(list(sanitized_params_flat.items())))
    mx.eval(controlled_text_model.parameters())

    original_shell_model_instance.language_model = controlled_text_model
    
    logger.info(f"Successfully loaded and wrapped '{model_name_or_path}' as ControlledGemma3TextModel.")
    return original_shell_model_instance, tokenizer
