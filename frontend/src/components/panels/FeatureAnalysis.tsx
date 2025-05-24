// components/panels/FeatureAnalysis.tsx
import React, { useState } from 'react';
import { Search, Plus, X, BarChart3, TrendingUp } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';

const FeatureAnalysis: React.FC = () => {
  const { model, analysis, analyzeFeature } = useAppStore();
  
  const [positivePrompts, setPositivePrompts] = useState(['']);
  const [negativePrompts, setNegativePrompts] = useState(['']);
  const [layersToAnalyze, setLayersToAnalyze] = useState('');
  const [controlPointsToAnalyze, setControlPointsToAnalyze] = useState<string[]>([]);
  const [metricType, setMetricType] = useState('cosine_distance');
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

  const toggleControlPoint = (point: string) => {
    setControlPointsToAnalyze(prev => 
      prev.includes(point) 
        ? prev.filter(p => p !== point)
        : [...prev, point]
    );
  };

  const handleAnalyze = async () => {
    const filteredPositive = positivePrompts.filter(p => p.trim());
    const filteredNegative = negativePrompts.filter(p => p.trim());

    if (filteredPositive.length === 0 || filteredNegative.length === 0) {
      return;
    }

    // Parse layers to analyze
    let layers: number[] | undefined;
    if (layersToAnalyze.trim()) {
      try {
        // Support formats like "0,5,10" or "0-10" or "0-5,15,20-25"
        const parts = layersToAnalyze.split(',').map(s => s.trim());
        layers = [];
        for (const part of parts) {
          if (part.includes('-')) {
            const [start, end] = part.split('-').map(n => parseInt(n.trim()));
            for (let i = start; i <= end; i++) {
              layers.push(i);
            }
          } else {
            layers.push(parseInt(part));
          }
        }
        // Remove duplicates and sort
        layers = [...new Set(layers)].sort((a, b) => a - b);
      } catch (error) {
        console.error('Invalid layers format:', error);
        layers = undefined;
      }
    }

    await analyzeFeature({
      positive_prompts: filteredPositive,
      negative_prompts: filteredNegative,
      layers_to_analyze: layers,
      control_points_to_analyze: controlPointsToAnalyze.length > 0 ? controlPointsToAnalyze : undefined,
      metric_type: metricType,
      average_over_tokens: averageOverTokens,
    });
  };

  const availableControlPoints = model.modelInfo?.available_control_points || [];
  const maxLayers = model.modelInfo?.num_layers || 0;

  if (!model.isLoaded || !model.modelInfo) {
    return (
      <div className="p-6 text-center">
        <Search className="w-12 h-12 text-secondary-500 mx-auto mb-4" />
        <p className="text-secondary-400">
          Load a model to analyze feature representations
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Search className="w-5 h-5 text-primary-400" />
          <h2 className="text-lg font-semibold text-secondary-100">
            Feature Analysis
          </h2>
        </div>

        <div className="space-y-4">
          {/* Analysis settings */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-secondary-300 mb-1">
                Layers to Analyze
              </label>
              <input
                type="text"
                value={layersToAnalyze}
                onChange={(e) => setLayersToAnalyze(e.target.value)}
                placeholder="e.g., 0-10, 20, 30-35 (empty = all)"
                className="input w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-secondary-300 mb-1">
                Metric
              </label>
              <select
                value={metricType}
                onChange={(e) => setMetricType(e.target.value)}
                className="input w-full"
              >
                <option value="cosine_distance">Cosine Distance</option>
                <option value="cosine_similarity">Cosine Similarity</option>
                <option value="l2_distance">L2 Distance</option>
              </select>
            </div>
          </div>

          {/* Control points selection */}
          <div>
            <label className="block text-sm font-medium text-secondary-300 mb-2">
              Control Points (empty = all)
            </label>
            <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto border border-secondary-700 rounded-lg p-3">
              {availableControlPoints.map(point => (
                <label key={point} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={controlPointsToAnalyze.includes(point)}
                    onChange={() => toggleControlPoint(point)}
                    className="rounded border-secondary-600 bg-secondary-700 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="text-xs text-secondary-300">
                    {point.replace(/_/g, ' ').substring(0, 20)}
                  </span>
                </label>
              ))}
            </div>
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
      <div className="flex-1 border-t border-secondary-700 flex flex-col">
        <div className="p-6 flex-1 overflow-y-auto">
          <div className="grid grid-cols-2 gap-6 h-full">
            {/* Feature-positive prompts */}
            <div className="flex flex-col">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-success-400">
                  Feature-Positive ({positivePrompts.filter(p => p.trim()).length})
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
                      placeholder="Enter prompts that exhibit the feature..."
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

            {/* Feature-negative prompts */}
            <div className="flex flex-col">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-error-400">
                  Feature-Negative ({negativePrompts.filter(p => p.trim()).length})
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
                      placeholder="Enter prompts that lack the feature..."
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
        </div>

        {/* Analyze button */}
        <div className="p-6 border-t border-secondary-700">
          <button
            onClick={handleAnalyze}
            disabled={
              analysis.isAnalyzing ||
              positivePrompts.filter(p => p.trim()).length === 0 ||
              negativePrompts.filter(p => p.trim()).length === 0
            }
            className="btn-primary w-full flex items-center justify-center space-x-2"
          >
            <BarChart3 className="w-4 h-4" />
            <span>{analysis.isAnalyzing ? 'Analyzing...' : 'Analyze Feature'}</span>
          </button>
        </div>

        {/* Results preview */}
        {analysis.lastAnalysis && analysis.lastAnalysis.results.length > 0 && (
          <div className="p-6 border-t border-secondary-700">
            <div className="flex items-center space-x-2 mb-3">
              <TrendingUp className="w-4 h-4 text-success-400" />
              <h3 className="text-sm font-medium text-secondary-300">
                Top Results ({analysis.lastAnalysis.results.length} total)
              </h3>
            </div>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {analysis.lastAnalysis.results.slice(0, 10).map((result, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-2 bg-secondary-700/30 rounded text-xs"
                >
                  <div>
                    <span className="text-secondary-200">
                      L{result.layer_idx}
                    </span>
                    <span className="text-secondary-400 ml-2">
                      {result.control_point.replace(/_/g, ' ').substring(0, 20)}
                    </span>
                  </div>
                  <div className="text-primary-400 font-mono">
                    {result.differentiation_score.toFixed(4)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FeatureAnalysis;
