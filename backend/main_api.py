# main_api.py
"""
FastAPI backend for MLX Control Vector Laboratory.
Provides REST API for multi-model LLM control vector experiments.
"""
import logging
import os
import sys
from typing import List, Dict, Optional, Any, Union
import json
import tempfile
from pathlib import Path

import mlx.core as mx
from fastapi import FastAPI, HTTPException, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our control modules
from model_loader import load_controlled_model, ModelLoadResult, get_supported_model_types
from control_utils import (
    derive_control_vector, load_control_vector, save_control_vector, 
    create_random_control_vector, clear_all_controls, get_model_hidden_size
)
from feature_analyzer import analyze_feature_activation, compare_feature_across_models
from control_core import get_control_points_for_model

# --- Pydantic Models for API ---

class ModelLoadRequest(BaseModel):
    model_path: str = Field(..., description="Path or Hugging Face ID of the MLX model.")
    tokenizer_config: Optional[Dict[str, Any]] = Field(default={}, description="Optional tokenizer configuration.")
    trust_remote_code: bool = Field(default=True, description="Whether to trust remote code.")
    force_model_type: Optional[str] = Field(None, description="Force specific model type detection.")

class ModelLoadResponse(BaseModel):
    status: str
    message: str
    model_type: str
    model_path: str
    num_layers: int
    hidden_size: int
    vocab_size: int
    is_controlled: bool
    architecture_info: Dict[str, Any]
    available_control_points: List[str]

class ControlVectorConfig(BaseModel):
    layer_idx: int = Field(..., ge=0, description="Layer index to apply the control.")
    control_point: str = Field(..., description="Control point name within the layer.")
    strength: float = Field(..., description="Strength of the control vector.")
    vector_source: Dict[str, Any] = Field(..., description="Vector source configuration.")

class ApplyControlsRequest(BaseModel):
    controls: List[ControlVectorConfig] = Field(..., description="List of control vector configurations.")

class DeriveVectorRequest(BaseModel):
    layer_idx: int = Field(..., ge=0, description="Layer index for vector derivation.")
    control_point: str = Field(..., description="Control point for derivation.")
    positive_prompts: List[str] = Field(..., description="Positive example prompts.")
    negative_prompts: List[str] = Field(..., description="Negative example prompts.")
    average_over_tokens: bool = Field(True, description="Whether to average activations over tokens.")
    vector_name: Optional[str] = Field(None, description="Name to save the derived vector.")

class DeriveVectorResponse(BaseModel):
    status: str
    message: str
    vector_shape: Optional[List[int]] = None
    vector_norm: Optional[float] = None
    saved_path: Optional[str] = None

class FeatureAnalysisRequest(BaseModel):
    positive_prompts: List[str] = Field(..., description="Feature-positive prompts.")
    negative_prompts: List[str] = Field(..., description="Feature-negative prompts.")
    layers_to_analyze: Optional[List[int]] = Field(None, description="Specific layers to analyze.")
    control_points_to_analyze: Optional[List[str]] = Field(None, description="Specific control points.")
    metric_type: str = Field("cosine_distance", description="Differentiation metric.")
    average_over_tokens: bool = Field(True, description="Average activations over tokens.")

class FeatureAnalysisResponse(BaseModel):
    status: str
    message: str
    model_type: str
    results: List[Dict[str, Any]]
    total_analyzed: int

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The prompt for text generation.")
    use_chat_template: bool = Field(default=False, description="Whether to apply chat template.")
    system_prompt: Optional[str] = Field(None, description="System prompt for chat template.")
    chat_template_args: Dict[str, Any] = Field(default={}, description="Chat template arguments.")
    max_tokens: int = Field(default=100, ge=1, le=2048, description="Maximum tokens to generate.")
    temp: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling.")
    stream: bool = Field(default=False, description="Stream generation tokens.")

class GenerateResponse(BaseModel):
    generated_text: str
    prompt_tokens: Optional[int] = None
    generation_tokens: Optional[int] = None

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    model_loaded: Optional[str] = None
    model_type: Optional[str] = None

# --- FastAPI App ---
app = FastAPI(
    title="MLX Control Vector Laboratory API",
    description="Multi-model LLM control vector experimentation platform",
    version="1.0.0"
)

# Add CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model_state = {
    "model_result": None,  # ModelLoadResult object
    "derived_vectors": {},  # {name: mx.array}
    "experiment_history": [],  # List of experiment results
}

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

# Helper functions
def _get_add_special_tokens_flag(raw_prompt_str: str, use_chat_template_flag: bool) -> bool:
    """Determine whether to add special tokens during tokenization."""
    if not model_state["model_result"]:
        return True
    tokenizer = model_state["model_result"].tokenizer
    if use_chat_template_flag and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        return False
    return True

def _process_raw_prompt_for_tokenization(
    raw_prompt_str: str, 
    use_chat_template_flag: bool, 
    system_prompt_str: Optional[str], 
    chat_template_args_dict: Dict[str, Any]
) -> str:
    """Process raw prompt with optional chat template."""
    if not model_state["model_result"]:
        return raw_prompt_str
        
    tokenizer = model_state["model_result"].tokenizer
    processed_prompt_str = raw_prompt_str
    
    if use_chat_template_flag and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = []
        if system_prompt_str:
            messages.append({"role": "system", "content": system_prompt_str})
        messages.append({"role": "user", "content": raw_prompt_str})
        
        try:
            processed_prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_args_dict
            )
        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            processed_prompt_str = raw_prompt_str
    
    return processed_prompt_str

# --- API Endpoints ---

@app.get("/", response_model=StatusResponse)
async def root():
    """API root endpoint with status information."""
    model_info = None
    if model_state["model_result"]:
        model_info = model_state["model_result"].model_path
        model_type = model_state["model_result"].model_type
    else:
        model_type = None
        
    return StatusResponse(
        status="success",
        message="MLX Control Vector Laboratory API is running",
        model_loaded=model_info,
        model_type=model_type
    )

@app.get("/supported_models", response_model=Dict[str, Any])
async def get_supported_models():
    """Get list of supported model architectures."""
    return {
        "supported_types": get_supported_model_types(),
        "model_architectures": {
            "gemma3": {"status": "fully_supported", "features": ["control_vectors", "feature_analysis"]},
            "llama": {"status": "placeholder", "features": ["basic_loading"]},
            "mixtral": {"status": "placeholder", "features": ["basic_loading"]}
        }
    }

@app.post("/load_model", response_model=ModelLoadResponse)
async def load_model_endpoint(request: ModelLoadRequest):
    """Load a model with control capabilities."""
    try:
        logger.info(f"Loading model: {request.model_path}")
        
        # Load the controlled model
        model_result = load_controlled_model(
            model_name_or_path=request.model_path,
            tokenizer_config=request.tokenizer_config,
            trust_remote_code=request.trust_remote_code,
            force_model_type=request.force_model_type
        )
        
        # Store in global state
        model_state["model_result"] = model_result
        model_state["derived_vectors"] = {}  # Clear previous vectors
        
        # Get available control points for this model type
        available_control_points = get_control_points_for_model(model_result.model_type)
        
        logger.info(f"Successfully loaded {model_result.model_type} model: {model_result.model_path}")
        
        return ModelLoadResponse(
            status="success",
            message=f"Model '{request.model_path}' loaded successfully",
            model_type=model_result.model_type,
            model_path=model_result.model_path,
            num_layers=model_result.num_layers,
            hidden_size=model_result.hidden_size,
            vocab_size=model_result.vocab_size,
            is_controlled=model_result.is_controlled,
            architecture_info=model_result.architecture_info,
            available_control_points=available_control_points
        )
        
    except Exception as e:
        logger.exception(f"Error loading model '{request.model_path}': {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/apply_controls", response_model=StatusResponse)
async def apply_controls_endpoint(request: ApplyControlsRequest):
    """Apply control vectors to the loaded model."""
    if not model_state["model_result"]:
        raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")
    
    model_result = model_state["model_result"]
    
    if not model_result.is_controlled:
        raise HTTPException(status_code=400, detail=f"Model type '{model_result.model_type}' does not support full control features yet.")
    
    try:
        # Clear existing controls
        clear_all_controls(model_result.model_shell)
        
        active_controls_applied = []
        hidden_size = model_result.hidden_size
        
        for control_spec in request.controls:
            control_vector = None
            vector_source = control_spec.vector_source
            
            # Validate layer index
            if not (0 <= control_spec.layer_idx < model_result.num_layers):
                logger.error(f"Invalid layer_idx {control_spec.layer_idx}")
                continue
            
            # Validate control point
            available_points = get_control_points_for_model(model_result.model_type)
            if control_spec.control_point not in available_points:
                logger.error(f"Invalid control_point '{control_spec.control_point}' for {model_result.model_type}")
                continue
            
            # Create control vector based on source type
            if vector_source["type"] == "random_positive":
                control_vector = create_random_control_vector((hidden_size,), "positive")
            elif vector_source["type"] == "random_negative":
                control_vector = create_random_control_vector((hidden_size,), "negative")
            elif vector_source["type"] == "load_from_file":
                file_path = vector_source.get("file_path")
                if file_path:
                    control_vector = load_control_vector(file_path, (hidden_size,))
            elif vector_source["type"] == "use_derived":
                vector_name = vector_source.get("vector_name")
                if vector_name and vector_name in model_state["derived_vectors"]:
                    control_vector = model_state["derived_vectors"][vector_name]
                else:
                    logger.error(f"Derived vector '{vector_name}' not found")
                    continue
            
            # Apply control vector
            if control_vector is not None:
                target_layer = model_result.model_shell.language_model.layers[control_spec.layer_idx]
                target_layer.add_control(
                    control_spec.control_point,
                    control_vector.astype(mx.float16),
                    control_spec.strength
                )
                active_controls_applied.append(
                    f"L{control_spec.layer_idx}|{control_spec.control_point}|S{control_spec.strength:.2f}"
                )
        
        if active_controls_applied:
            message = f"Applied {len(active_controls_applied)} controls: {', '.join(active_controls_applied)}"
        else:
            message = "No valid controls were applied"
        
        return StatusResponse(status="success", message=message)
        
    except Exception as e:
        logger.exception(f"Error applying controls: {e}")
        raise HTTPException(status_code=500, detail=f"Error applying controls: {str(e)}")

@app.post("/derive_vector", response_model=DeriveVectorResponse)
async def derive_vector_endpoint(request: DeriveVectorRequest):
    """Derive a control vector from positive and negative prompts."""
    if not model_state["model_result"]:
        raise HTTPException(status_code=400, detail="No model loaded.")
    
    model_result = model_state["model_result"]
    
    if not model_result.is_controlled:
        raise HTTPException(status_code=400, detail=f"Model type '{model_result.model_type}' does not support vector derivation yet.")
    
    try:
        # Process prompts
        positive_processed = [
            _process_raw_prompt_for_tokenization(p, False, None, {}) 
            for p in request.positive_prompts
        ]
        negative_processed = [
            _process_raw_prompt_for_tokenization(p, False, None, {})
            for p in request.negative_prompts
        ]
        
        # Derive control vector
        control_vector = derive_control_vector(
            model_result.model_shell,
            model_result.tokenizer,
            positive_processed,
            negative_processed,
            request.layer_idx,
            request.control_point,
            request.average_over_tokens,
            model_result.model_type
        )
        
        if control_vector is None:
            raise HTTPException(status_code=500, detail="Failed to derive control vector")
        
        # Store derived vector
        vector_name = request.vector_name or f"derived_L{request.layer_idx}_{request.control_point}"
        model_state["derived_vectors"][vector_name] = control_vector
        
        # Optionally save to file
        saved_path = None
        if request.vector_name:
            save_dir = Path("derived_vectors")
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"{vector_name}.npz"
            
            metadata = {
                "model_type": model_result.model_type,
                "layer_idx": request.layer_idx,
                "control_point": request.control_point,
                "positive_prompts": request.positive_prompts,
                "negative_prompts": request.negative_prompts
            }
            
            if save_control_vector(control_vector, str(save_path), metadata):
                saved_path = str(save_path)
        
        return DeriveVectorResponse(
            status="success",
            message=f"Derived control vector '{vector_name}' successfully",
            vector_shape=list(control_vector.shape),
            vector_norm=float(mx.linalg.norm(control_vector).item()),
            saved_path=saved_path
        )
        
    except Exception as e:
        logger.exception(f"Error deriving control vector: {e}")
        raise HTTPException(status_code=500, detail=f"Error deriving control vector: {str(e)}")

@app.post("/analyze_feature", response_model=FeatureAnalysisResponse)
async def analyze_feature_endpoint(request: FeatureAnalysisRequest):
    """Analyze where a feature is encoded in the model."""
    if not model_state["model_result"]:
        raise HTTPException(status_code=400, detail="No model loaded.")
    
    model_result = model_state["model_result"]
    
    if not model_result.is_controlled:
        raise HTTPException(status_code=400, detail=f"Model type '{model_result.model_type}' does not support feature analysis yet.")
    
    try:
        # Set defaults if not specified
        layers_to_analyze = request.layers_to_analyze or list(range(model_result.num_layers))
        control_points_to_analyze = request.control_points_to_analyze or get_control_points_for_model(model_result.model_type)
        
        # Run feature analysis
        results = analyze_feature_activation(
            model_result.model_shell,
            model_result.tokenizer,
            request.positive_prompts,
            request.negative_prompts,
            layers_to_analyze,
            control_points_to_analyze,
            lambda p: _process_raw_prompt_for_tokenization(p, False, None, {}),
            lambda p: _get_add_special_tokens_flag(p, False),
            request.metric_type,
            request.average_over_tokens,
            model_result.model_type
        )
        
        return FeatureAnalysisResponse(
            status="success",
            message=f"Analyzed {len(results)} layer/control-point combinations",
            model_type=model_result.model_type,
            results=results,
            total_analyzed=len(results)
        )
        
    except Exception as e:
        logger.exception(f"Error analyzing feature: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing feature: {str(e)}")

@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest):
    """Generate text with the current model state."""
    if not model_state["model_result"]:
        raise HTTPException(status_code=400, detail="No model loaded.")
    
    model_result = model_state["model_result"]
    
    try:
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm import generate as generate_text_mlx
        
        # Process prompt
        processed_prompt = _process_raw_prompt_for_tokenization(
            request.prompt,
            request.use_chat_template,
            request.system_prompt,
            request.chat_template_args
        )
        
        # Tokenize
        add_special_tokens = _get_add_special_tokens_flag(request.prompt, request.use_chat_template)
        if hasattr(model_result.tokenizer, 'encode'):
            prompt_tokens = model_result.tokenizer.encode(processed_prompt, add_special_tokens=add_special_tokens)
        else:
            tokenized = model_result.tokenizer(processed_prompt, add_special_tokens=add_special_tokens, return_tensors="np")
            prompt_tokens = tokenized["input_ids"][0]
        
        if not prompt_tokens:
            raise HTTPException(status_code=400, detail="Empty tokenization result")
        
        prompt_tokens_mx = mx.array([prompt_tokens])  # Add batch dimension
        
        # Create sampler
        sampler = make_sampler(temp=request.temp, top_p=request.top_p)
        
        # Generate text
        logger.info(f"Generating text for prompt: '{request.prompt[:60]}...'")
        
        generated_text = generate_text_mlx(
            model_result.model_shell,
            model_result.tokenizer,
            prompt=prompt_tokens_mx,
            max_tokens=request.max_tokens,
            sampler=sampler,
            verbose=False
        )
        
        # Calculate token counts (approximate)
        if hasattr(model_result.tokenizer, 'encode'):
            generation_tokens = len(model_result.tokenizer.encode(generated_text, add_special_tokens=False))
        else:
            generation_tokens = len(generated_text.split())  # Rough approximation
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt_tokens=len(prompt_tokens),
            generation_tokens=generation_tokens
        )
        
    except Exception as e:
        logger.exception(f"Error during text generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error during text generation: {str(e)}")

@app.post("/clear_controls", response_model=StatusResponse)
async def clear_controls_endpoint():
    """Clear all active control vectors."""
    if not model_state["model_result"]:
        raise HTTPException(status_code=400, detail="No model loaded.")
    
    try:
        clear_all_controls(model_state["model_result"].model_shell)
        return StatusResponse(status="success", message="All controls cleared")
    except Exception as e:
        logger.exception(f"Error clearing controls: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing controls: {str(e)}")

@app.get("/derived_vectors", response_model=Dict[str, Any])
async def get_derived_vectors():
    """Get list of available derived vectors."""
    vectors_info = {}
    for name, vector in model_state["derived_vectors"].items():
        vectors_info[name] = {
            "shape": list(vector.shape),
            "norm": float(mx.linalg.norm(vector).item()),
            "dtype": str(vector.dtype)
        }
    
    return {
        "status": "success",
        "derived_vectors": vectors_info,
        "count": len(vectors_info)
    }

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "main_api:app",
        host="127.0.0.1",
        port=8008,
        reload=True,
        log_level="info"
    )
