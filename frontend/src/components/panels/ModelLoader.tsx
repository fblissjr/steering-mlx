// components/panels/ModelLoader.tsx
import React, { useState } from 'react';
import { Upload, Download, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';

const ModelLoader: React.FC = () => {
  const { model, loadModel } = useAppStore();
  const [modelPath, setModelPath] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [forceModelType, setForceModelType] = useState('');

  const handleLoadModel = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modelPath.trim()) return;

    await loadModel(modelPath.trim(), {
      force_model_type: forceModelType || undefined,
    });
  };

  const popularModels = [
    {
      name: 'Gemma 2B',
      path: 'mlx-community/gemma-2b-it-4bit',
      description: 'Small, fast model for testing',
      type: 'gemma3',
    },
    {
      name: 'Gemma 7B',
      path: 'mlx-community/gemma-7b-it-4bit',
      description: 'Balanced performance and quality',
      type: 'gemma3',
    },
    {
      name: 'Llama 3.2 3B',
      path: 'mlx-community/Llama-3.2-3B-Instruct-4bit',
      description: 'Efficient instruction-following',
      type: 'llama',
    },
    {
      name: 'Mixtral 8x7B',
      path: 'mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit',
      description: 'Mixture of Experts model',
      type: 'mixtral',
    },
  ];

  return (
    <div className="p-6">
      <div className="flex items-center space-x-2 mb-4">
        <Upload className="w-5 h-5 text-primary-400" />
        <h2 className="text-lg font-semibold text-secondary-100">
          Model Loader
        </h2>
      </div>

      {/* Current model status */}
      {model.isLoaded && model.modelInfo && (
        <div className="mb-4 p-3 bg-success-900/20 border border-success-700 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <CheckCircle className="w-4 h-4 text-success-400" />
            <span className="text-sm font-medium text-success-400">
              Model Loaded
            </span>
          </div>
          <div className="text-xs text-secondary-300">
            <div>{model.modelInfo.model_type.toUpperCase()} • {model.modelInfo.num_layers} layers</div>
            <div className="text-secondary-400 truncate mt-1">
              {model.modelInfo.model_path}
            </div>
          </div>
        </div>
      )}

      {/* Error display */}
      {model.error && (
        <div className="mb-4 p-3 bg-error-900/20 border border-error-700 rounded-lg">
          <div className="flex items-center space-x-2 mb-1">
            <AlertCircle className="w-4 h-4 text-error-400" />
            <span className="text-sm font-medium text-error-400">
              Loading Error
            </span>
          </div>
          <div className="text-xs text-error-300">
            {model.error}
          </div>
        </div>
      )}

      {/* Load form */}
      <form onSubmit={handleLoadModel} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-secondary-300 mb-2">
            Model Path or HuggingFace ID
          </label>
          <input
            type="text"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            placeholder="e.g., mlx-community/gemma-2b-it-4bit"
            className="input w-full"
            disabled={model.isLoading}
          />
          <p className="text-xs text-secondary-400 mt-1">
            Enter a HuggingFace model ID or local path
          </p>
        </div>

        {/* Advanced options */}
        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-sm text-secondary-400 hover:text-secondary-200 transition-colors"
          >
            {showAdvanced ? '▼' : '▶'} Advanced Options
          </button>
          
          {showAdvanced && (
            <div className="mt-3 space-y-3 p-3 bg-secondary-900/50 rounded-lg border border-secondary-700">
              <div>
                <label className="block text-sm font-medium text-secondary-300 mb-1">
                  Force Model Type
                </label>
                <select
                  value={forceModelType}
                  onChange={(e) => setForceModelType(e.target.value)}
                  className="input w-full"
                >
                  <option value="">Auto-detect</option>
                  <option value="gemma3">Gemma 3</option>
                  <option value="llama">Llama</option>
                  <option value="mixtral">Mixtral</option>
                </select>
              </div>
            </div>
          )}
        </div>

        <button
          type="submit"
          disabled={model.isLoading || !modelPath.trim()}
          className="btn-primary w-full flex items-center justify-center space-x-2"
        >
          {model.isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Loading...</span>
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              <span>Load Model</span>
            </>
          )}
        </button>
      </form>

      {/* Popular models */}
      <div className="mt-6">
        <h3 className="text-sm font-medium text-secondary-300 mb-3">
          Popular Models
        </h3>
        <div className="space-y-2">
          {popularModels.map((model, index) => (
            <button
              key={index}
              onClick={() => setModelPath(model.path)}
              className="w-full p-3 text-left bg-secondary-700/50 hover:bg-secondary-700 rounded-lg transition-colors border border-secondary-600"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-secondary-200">
                  {model.name}
                </span>
                <span className={`text-xs px-2 py-1 rounded ${
                  model.type === 'gemma3' 
                    ? 'bg-primary-600 text-white' 
                    : 'bg-secondary-600 text-secondary-200'
                }`}>
                  {model.type}
                </span>
              </div>
              <p className="text-xs text-secondary-400">
                {model.description}
              </p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelLoader;
