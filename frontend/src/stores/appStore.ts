// stores/appStore.ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { apiService, ModelLoadResponse, ControlVectorConfig } from '../services/api';
import toast from 'react-hot-toast';

export interface ModelState {
  isLoaded: boolean;
  isLoading: boolean;
  modelInfo: ModelLoadResponse | null;
  error: string | null;
}

export interface ControlState {
  activeControls: ControlVectorConfig[];
  isApplying: boolean;
  derivedVectors: Record<string, any>;
  error: string | null;
}

export interface GenerationState {
  isGenerating: boolean;
  lastGeneration: {
    prompt: string;
    response: string;
    timestamp: number;
  } | null;
  error: string | null;
}

export interface AnalysisState {
  isAnalyzing: boolean;
  lastAnalysis: {
    results: any[];
    timestamp: number;
    model_type: string;
  } | null;
  error: string | null;
}

export interface UIState {
  activeTab: 'controls' | 'derivation' | 'analysis' | 'generation';
  sidebarCollapsed: boolean;
  selectedLayer: number | null;
  selectedControlPoint: string | null;
  traceMode: boolean;
}

interface AppState {
  // Model state
  model: ModelState;
  
  // Control state
  controls: ControlState;
  
  // Generation state
  generation: GenerationState;
  
  // Analysis state
  analysis: AnalysisState;
  
  // UI state
  ui: UIState;
  
  // Actions
  loadModel: (modelPath: string, options?: Partial<{
    tokenizer_config: Record<string, any>;
    trust_remote_code: boolean;
    force_model_type: string;
  }>) => Promise<void>;
  
  addControl: (control: ControlVectorConfig) => void;
  removeControl: (index: number) => void;
  applyControls: () => Promise<void>;
  clearControls: () => Promise<void>;
  
  deriveVector: (request: {
    layer_idx: number;
    control_point: string;
    positive_prompts: string[];
    negative_prompts: string[];
    vector_name?: string;
  }) => Promise<void>;
  
  generate: (request: {
    prompt: string;
    max_tokens?: number;
    temp?: number;
    top_p?: number;
  }) => Promise<void>;
  
  analyzeFeature: (request: {
    positive_prompts: string[];
    negative_prompts: string[];
    layers_to_analyze?: number[];
    control_points_to_analyze?: string[];
  }) => Promise<void>;
  
  setActiveTab: (tab: UIState['activeTab']) => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setSelectedLayer: (layer: number | null) => void;
  setSelectedControlPoint: (point: string | null) => void;
  setTraceMode: (enabled: boolean) => void;
  
  reset: () => void;
}

const initialState = {
  model: {
    isLoaded: false,
    isLoading: false,
    modelInfo: null,
    error: null,
  },
  controls: {
    activeControls: [],
    isApplying: false,
    derivedVectors: {},
    error: null,
  },
  generation: {
    isGenerating: false,
    lastGeneration: null,
    error: null,
  },
  analysis: {
    isAnalyzing: false,
    lastAnalysis: null,
    error: null,
  },
  ui: {
    activeTab: 'controls' as const,
    sidebarCollapsed: false,
    selectedLayer: null,
    selectedControlPoint: null,
    traceMode: false,
  },
};

export const useAppStore = create<AppState>()(
  devtools(
    (set, get) => ({
      ...initialState,
      
      loadModel: async (modelPath, options = {}) => {
        const { model } = get();
        if (model.isLoading) return;
        
        set((state) => ({
          model: { ...state.model, isLoading: true, error: null }
        }));
        
        try {
          const modelInfo = await apiService.loadModel({
            model_path: modelPath,
            tokenizer_config: options.tokenizer_config || {},
            trust_remote_code: options.trust_remote_code ?? true,
            force_model_type: options.force_model_type,
          });
          
          set((state) => ({
            model: {
              ...state.model,
              isLoaded: true,
              isLoading: false,
              modelInfo,
              error: null,
            },
            controls: {
              ...state.controls,
              activeControls: [],
              derivedVectors: {},
            }
          }));
          
          toast.success(`Model ${modelInfo.model_type} loaded successfully!`);
          
          // Load derived vectors
          try {
            const derivedVectors = await apiService.getDerivedVectors();
            set((state) => ({
              controls: { ...state.controls, derivedVectors: derivedVectors.derived_vectors || {} }
            }));
          } catch (error) {
            console.warn('Failed to load derived vectors:', error);
          }
          
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || error.message || 'Failed to load model';
          set((state) => ({
            model: { ...state.model, isLoading: false, error: errorMessage }
          }));
          toast.error(`Failed to load model: ${errorMessage}`);
        }
      },
      
      addControl: (control) => {
        set((state) => ({
          controls: {
            ...state.controls,
            activeControls: [...state.controls.activeControls, control]
          }
        }));
      },
      
      removeControl: (index) => {
        set((state) => ({
          controls: {
            ...state.controls,
            activeControls: state.controls.activeControls.filter((_, i) => i !== index)
          }
        }));
      },
      
      applyControls: async () => {
        const { controls } = get();
        if (controls.isApplying || controls.activeControls.length === 0) return;
        
        set((state) => ({
          controls: { ...state.controls, isApplying: true, error: null }
        }));
        
        try {
          await apiService.applyControls({ controls: controls.activeControls });
          set((state) => ({
            controls: { ...state.controls, isApplying: false, error: null }
          }));
          toast.success(`Applied ${controls.activeControls.length} control(s)!`);
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || error.message || 'Failed to apply controls';
          set((state) => ({
            controls: { ...state.controls, isApplying: false, error: errorMessage }
          }));
          toast.error(`Failed to apply controls: ${errorMessage}`);
        }
      },
      
      clearControls: async () => {
        try {
          await apiService.clearControls();
          set((state) => ({
            controls: { ...state.controls, activeControls: [], error: null }
          }));
          toast.success('All controls cleared!');
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || error.message || 'Failed to clear controls';
          toast.error(`Failed to clear controls: ${errorMessage}`);
        }
      },
      
      deriveVector: async (request) => {
        set((state) => ({
          controls: { ...state.controls, error: null }
        }));
        
        try {
          const result = await apiService.deriveVector(request);
          
          // Refresh derived vectors list
          const derivedVectors = await apiService.getDerivedVectors();
          set((state) => ({
            controls: { ...state.controls, derivedVectors: derivedVectors.derived_vectors || {} }
          }));
          
          toast.success(`Vector derived successfully! Norm: ${result.vector_norm?.toFixed(4)}`);
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || error.message || 'Failed to derive vector';
          set((state) => ({
            controls: { ...state.controls, error: errorMessage }
          }));
          toast.error(`Failed to derive vector: ${errorMessage}`);
        }
      },
      
      generate: async (request) => {
        const { generation } = get();
        if (generation.isGenerating) return;
        
        set((state) => ({
          generation: { ...state.generation, isGenerating: true, error: null }
        }));
        
        try {
          const result = await apiService.generate(request);
          set((state) => ({
            generation: {
              ...state.generation,
              isGenerating: false,
              lastGeneration: {
                prompt: request.prompt,
                response: result.generated_text,
                timestamp: Date.now(),
              },
              error: null,
            }
          }));
          toast.success('Text generated successfully!');
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || error.message || 'Failed to generate text';
          set((state) => ({
            generation: { ...state.generation, isGenerating: false, error: errorMessage }
          }));
          toast.error(`Generation failed: ${errorMessage}`);
        }
      },
      
      analyzeFeature: async (request) => {
        const { analysis } = get();
        if (analysis.isAnalyzing) return;
        
        set((state) => ({
          analysis: { ...state.analysis, isAnalyzing: true, error: null }
        }));
        
        try {
          const result = await apiService.analyzeFeature(request);
          set((state) => ({
            analysis: {
              ...state.analysis,
              isAnalyzing: false,
              lastAnalysis: {
                results: result.results,
                timestamp: Date.now(),
                model_type: result.model_type,
              },
              error: null,
            }
          }));
          toast.success(`Analysis completed! Found ${result.results.length} results.`);
        } catch (error: any) {
          const errorMessage = error.response?.data?.detail || error.message || 'Failed to analyze feature';
          set((state) => ({
            analysis: { ...state.analysis, isAnalyzing: false, error: errorMessage }
          }));
          toast.error(`Analysis failed: ${errorMessage}`);
        }
      },
      
      setActiveTab: (tab) => {
        set((state) => ({
          ui: { ...state.ui, activeTab: tab }
        }));
      },
      
      setSidebarCollapsed: (collapsed) => {
        set((state) => ({
          ui: { ...state.ui, sidebarCollapsed: collapsed }
        }));
      },
      
      setSelectedLayer: (layer) => {
        set((state) => ({
          ui: { ...state.ui, selectedLayer: layer }
        }));
      },
      
      setSelectedControlPoint: (point) => {
        set((state) => ({
          ui: { ...state.ui, selectedControlPoint: point }
        }));
      },
      
      setTraceMode: (enabled) => {
        set((state) => ({
          ui: { ...state.ui, traceMode: enabled }
        }));
      },
      
      reset: () => {
        set(initialState);
      },
    }),
    {
      name: 'mlx-control-vector-lab',
    }
  )
);

export default useAppStore;
