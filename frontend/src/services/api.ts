// services/api.ts
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`🔄 API Request: ${config.method?.toUpperCase()} ${config.url}`, config.data);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.config.method?.toUpperCase()} ${response.config.url}`, response.data);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Type definitions
export interface ModelLoadRequest {
  model_path: string;
  tokenizer_config?: Record<string, any>;
  trust_remote_code?: boolean;
  force_model_type?: string;
}

export interface ModelLoadResponse {
  status: string;
  message: string;
  model_type: string;
  model_path: string;
  num_layers: number;
  hidden_size: number;
  vocab_size: number;
  is_controlled: boolean;
  architecture_info: Record<string, any>;
  available_control_points: string[];
}

export interface ControlVectorConfig {
  layer_idx: number;
  control_point: string;
  strength: number;
  vector_source: Record<string, any>;
}

export interface ApplyControlsRequest {
  controls: ControlVectorConfig[];
}

export interface DeriveVectorRequest {
  layer_idx: number;
  control_point: string;
  positive_prompts: string[];
  negative_prompts: string[];
  average_over_tokens?: boolean;
  vector_name?: string;
}

export interface DeriveVectorResponse {
  status: string;
  message: string;
  vector_shape?: number[];
  vector_norm?: number;
  saved_path?: string;
}

export interface FeatureAnalysisRequest {
  positive_prompts: string[];
  negative_prompts: string[];
  layers_to_analyze?: number[];
  control_points_to_analyze?: string[];
  metric_type?: string;
  average_over_tokens?: boolean;
}

export interface FeatureAnalysisResponse {
  status: string;
  message: string;
  model_type: string;
  results: Array<{
    model_type: string;
    layer_idx: number;
    control_point: string;
    metric_type: string;
    differentiation_score: number;
    norm_pos_vector: number;
    norm_neg_vector: number;
  }>;
  total_analyzed: number;
}

export interface GenerateRequest {
  prompt: string;
  use_chat_template?: boolean;
  system_prompt?: string;
  chat_template_args?: Record<string, any>;
  max_tokens?: number;
  temp?: number;
  top_p?: number;
  stream?: boolean;
}

export interface GenerateResponse {
  generated_text: string;
  prompt_tokens?: number;
  generation_tokens?: number;
}

export interface StatusResponse {
  status: string;
  message?: string;
  model_loaded?: string;
  model_type?: string;
}

// API functions
export const apiService = {
  // System endpoints
  async getStatus(): Promise<StatusResponse> {
    const response = await api.get('/');
    return response.data;
  },

  async getSupportedModels(): Promise<Record<string, any>> {
    const response = await api.get('/supported_models');
    return response.data;
  },

  // Model management
  async loadModel(request: ModelLoadRequest): Promise<ModelLoadResponse> {
    const response = await api.post('/load_model', request);
    return response.data;
  },

  // Control vector operations
  async applyControls(request: ApplyControlsRequest): Promise<StatusResponse> {
    const response = await api.post('/apply_controls', request);
    return response.data;
  },

  async clearControls(): Promise<StatusResponse> {
    const response = await api.post('/clear_controls');
    return response.data;
  },

  async deriveVector(request: DeriveVectorRequest): Promise<DeriveVectorResponse> {
    const response = await api.post('/derive_vector', request);
    return response.data;
  },

  async getDerivedVectors(): Promise<Record<string, any>> {
    const response = await api.get('/derived_vectors');
    return response.data;
  },

  // Feature analysis
  async analyzeFeature(request: FeatureAnalysisRequest): Promise<FeatureAnalysisResponse> {
    const response = await api.post('/analyze_feature', request);
    return response.data;
  },

  // Text generation
  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    const response = await api.post('/generate', request);
    return response.data;
  },
};

export default apiService;
