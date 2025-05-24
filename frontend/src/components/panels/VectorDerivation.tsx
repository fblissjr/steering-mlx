// components/panels/VectorDerivation.tsx
import React, { useState } from 'react';
import { FlaskConical, Plus, X, ArrowRight, Save } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';

const VectorDerivation: React.FC = () => {
  const { model, deriveVector } = useAppStore();
  
  const [layerIdx, setLayerIdx] = useState(0);
  const [controlPoint, setControlPoint] = useState('');
  const [vectorName, setVectorName] = useState('');
  const [positivePrompts, setPositivePrompts] = useState(['']);
  const [negativePrompts, setNegativePrompts] = useState(['']);
  const [averageOverTokens, setAverageOverTokens] = useState(true);

  const addPrompt = (type: 'positive' | 'negative') => {
    if (type === 'positive') {
      setPositivePrompts([...positivePrompts, '']);
    } else {
      setNegativePrompts([...negativePrompts, '']);
    }
  };

  const removePrompt = (type: 'positive' | 'negative', index: number) => {
    if (type === 'positive') {
      setPositivePrompts(positivePrompts.filter((_, i) => i !== index));
    } else {
      setNegativePrompts(negativePrompts.filter((_, i) => i !== index));
    }
  };

  const updatePrompt = (type: 'positive' | 'negative', index: number, value: string) => {
    if (type === 'positive') {
      const updated = [...positivePrompts];
      updated[index] = value;
      setPositivePrompts(updated);
    } else {
      const updated = [...negativePrompts];
      updated[index] = value;
      setNegativePrompts(updated);
    }
  };

  const handleDerive = async () => {
    if (!controlPoint || positivePrompts.filter(p => p.trim()).length === 0 || 
        negativePrompts.filter(p => p.trim()).length === 0) {
      return;
    }

    const filteredPositive = positivePrompts.filter(p => p.trim());
    const filteredNegative = negativePrompts.filter(p => p.trim());

    await deriveVector({
      layer_idx: layerIdx,
      control_point: controlPoint,
      positive_prompts: filteredPositive,
      negative_prompts: filteredNegative,
      vector_name: vectorName.trim() || undefined,
    });
  };

  const availableControlPoints = model.modelInfo?.available_control_points || [];
  const maxLayers = model.modelInfo?.num_layers || 0;

  if (!model.isLoaded || !model.modelInfo) {
    return (
      <div className="p-6 text-center">
        <FlaskConical className="w-12 h-12 text-secondary-500 mx-auto mb-4" />
        <p className="text-secondary-400">
          Load a model to derive control vectors
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-6">
        <div className="flex items-center space-x-2 mb-4">
          <FlaskConical className="w-5 h-5 text-primary-400" />
          <h2 className="text-lg font-semibold text-secondary-100">
            Vector Derivation
          </h2>
        </div>

        <div className="space-y-4">
          {/* Target settings */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-secondary-300 mb-1">
                Layer
              </label>
              <select
                value={layerIdx}
                onChange={(e) => setLayerIdx(parseInt(e.target.value))}
                className="input w-full"
              >
                {Array.from({ length: maxLayers }, (_, i) => (
                  <option key={i} value={i}>Layer {i}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-secondary-300 mb-1">
                Control Point
              </label>
              <select
                value={controlPoint}
                onChange={(e) => setControlPoint(e.target.value)}
                className="input w-full"
              >
                <option value="">Select...</option>
                {availableControlPoints.map(point => (
                  <option key={point} value={point}>
                    {point.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Vector name */}
          <div>
            <label className="block text-sm font-medium text-secondary-300 mb-1">
              Vector Name (optional)
            </label>
            <input
              type="text"
              value={vectorName}
              onChange={(e) => setVectorName(e.target.value)}
              placeholder="e.g., formality_vector"
              className="input w-full"
            />
          </div>

          {/* Settings */}
          <div className="flex items-center space-x-3">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={averageOverTokens}
                onChange={(e) => setAverageOverTokens(e.target.checked)}
                className="rounded border-secondary-600 bg-secondary-700 text-primary-600 focus:ring-primary-500"
              />
              <span className="text-sm text-secondary-300">
                Average over tokens
              </span>
            </label>
          </div>
        </div>
      </div>

      {/* Prompts section */}
      <div className="flex-1 p-6 border-t border-secondary-700 overflow-y-auto">
        <div className="grid grid-cols-2 gap-6 h-full">
          {/* Positive prompts */}
          <div className="flex flex-col">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-success-400">
                Positive Prompts ({positivePrompts.filter(p => p.trim()).length})
              </h3>
              <button
                onClick={() => addPrompt('positive')}
                className="text-success-400 hover:text-success-300 transition-colors"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>
            <div className="flex-1 space-y-3 overflow-y-auto">
              {positivePrompts.map((prompt, index) => (
                <div key={index} className="relative">
                  <textarea
                    value={prompt}
                    onChange={(e) => updatePrompt('positive', index, e.target.value)}
                    placeholder="Enter a positive example prompt..."
                    className="input w-full h-20 resize-none pr-8"
                  />
                  {positivePrompts.length > 1 && (
                    <button
                      onClick={() => removePrompt('positive', index)}
                      className="absolute top-2 right-2 text-error-400 hover:text-error-300 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Arrow */}
          <div className="flex items-center justify-center">
            <ArrowRight className="w-6 h-6 text-secondary-500" />
          </div>

          {/* Negative prompts */}
          <div className="flex flex-col">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-error-400">
                Negative Prompts ({negativePrompts.filter(p => p.trim()).length})
              </h3>
              <button
                onClick={() => addPrompt('negative')}
                className="text-error-400 hover:text-error-300 transition-colors"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>
            <div className="flex-1 space-y-3 overflow-y-auto">
              {negativePrompts.map((prompt, index) => (
                <div key={index} className="relative">
                  <textarea
                    value={prompt}
                    onChange={(e) => updatePrompt('negative', index, e.target.value)}
                    placeholder="Enter a negative example prompt..."
                    className="input w-full h-20 resize-none pr-8"
                  />
                  {negativePrompts.length > 1 && (
                    <button
                      onClick={() => removePrompt('negative', index)}
                      className="absolute top-2 right-2 text-error-400 hover:text-error-300 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Derive button */}
        <div className="mt-6 pt-4 border-t border-secondary-700">
          <button
            onClick={handleDerive}
            disabled={
              !controlPoint || 
              positivePrompts.filter(p => p.trim()).length === 0 ||
              negativePrompts.filter(p => p.trim()).length === 0
            }
            className="btn-primary w-full flex items-center justify-center space-x-2"
          >
            <Save className="w-4 h-4" />
            <span>Derive Vector</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default VectorDerivation;
