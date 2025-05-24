// components/panels/ControlPanel.tsx
import React, { useState, useEffect } from 'react';
import { Plus, X, Settings, Play, Trash2, Target } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';
import { ControlVectorConfig } from '../../services/api';

const ControlPanel: React.FC = () => {
  const { 
    model, 
    controls, 
    ui,
    addControl, 
    removeControl, 
    applyControls, 
    clearControls,
    setSelectedLayer,
    setSelectedControlPoint
  } = useAppStore();

  const [newControl, setNewControl] = useState<Partial<ControlVectorConfig>>({
    layer_idx: ui.selectedLayer ?? 0,
    control_point: ui.selectedControlPoint ?? '',
    strength: 1.0,
    vector_source: { type: 'random_positive' }
  });

  // Update form when selections change from model visualization
  useEffect(() => {
    if (ui.selectedLayer !== null) {
      setNewControl(prev => ({ ...prev, layer_idx: ui.selectedLayer! }));
    }
    if (ui.selectedControlPoint) {
      setNewControl(prev => ({ ...prev, control_point: ui.selectedControlPoint! }));
    }
  }, [ui.selectedLayer, ui.selectedControlPoint]);

  const handleAddControl = () => {
    if (!model.modelInfo || !newControl.control_point) return;

    const control: ControlVectorConfig = {
      layer_idx: newControl.layer_idx ?? 0,
      control_point: newControl.control_point,
      strength: newControl.strength ?? 1.0,
      vector_source: newControl.vector_source ?? { type: 'random_positive' }
    };

    addControl(control);
    
    // Reset form but keep layer/point selection
    setNewControl(prev => ({
      ...prev,
      strength: 1.0,
      vector_source: { type: 'random_positive' }
    }));
  };

  const updateVectorSource = (type: string, additional: Record<string, any> = {}) => {
    setNewControl(prev => ({
      ...prev,
      vector_source: { type, ...additional }
    }));
  };

  const availableControlPoints = model.modelInfo?.available_control_points || [];
  const maxLayers = model.modelInfo?.num_layers || 0;

  if (!model.isLoaded || !model.modelInfo) {
    return (
      <div className="p-6 text-center">
        <Settings className="w-12 h-12 text-secondary-500 mx-auto mb-4" />
        <p className="text-secondary-400">
          Load a model to configure control vectors
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-6 border-b border-secondary-700">
        <div className="flex items-center space-x-2 mb-4">
          <Settings className="w-5 h-5 text-primary-400" />
          <h2 className="text-lg font-semibold text-secondary-100">
            Control Vectors
          </h2>
        </div>

        {/* Add new control form */}
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-secondary-300 mb-1">
                Layer
              </label>
              <select
                value={newControl.layer_idx ?? 0}
                onChange={(e) => {
                  const layer = parseInt(e.target.value);
                  setNewControl(prev => ({ ...prev, layer_idx: layer }));
                  setSelectedLayer(layer);
                }}
                className="input w-full"
              >
                {Array.from({ length: maxLayers }, (_, i) => (
                  <option key={i} value={i}>Layer {i}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-secondary-300 mb-1">
                Strength
              </label>
              <input
                type="number"
                step="0.1"
                value={newControl.strength ?? 1.0}
                onChange={(e) => setNewControl(prev => ({ 
                  ...prev, 
                  strength: parseFloat(e.target.value) || 0 
                }))}
                className="input w-full"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-secondary-300 mb-1">
              Control Point
            </label>
            <select
              value={newControl.control_point ?? ''}
              onChange={(e) => {
                setNewControl(prev => ({ ...prev, control_point: e.target.value }));
                setSelectedControlPoint(e.target.value);
              }}
              className="input w-full"
            >
              <option value="">Select control point...</option>
              {availableControlPoints.map(point => (
                <option key={point} value={point}>
                  {point.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-secondary-300 mb-1">
              Vector Source
            </label>
            <select
              value={newControl.vector_source?.type ?? 'random_positive'}
              onChange={(e) => updateVectorSource(e.target.value)}
              className="input w-full"
            >
              <option value="random_positive">Random Positive</option>
              <option value="random_negative">Random Negative</option>
              <option value="use_derived">Use Derived Vector</option>
              <option value="load_from_file">Load from File</option>
            </select>

            {/* Additional options based on vector source */}
            {newControl.vector_source?.type === 'use_derived' && (
              <select
                className="input w-full mt-2"
                onChange={(e) => updateVectorSource('use_derived', { vector_name: e.target.value })}
              >
                <option value="">Select derived vector...</option>
                {Object.keys(controls.derivedVectors).map(name => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            )}

            {newControl.vector_source?.type === 'load_from_file' && (
              <input
                type="text"
                placeholder="Path to .npy or .npz file"
                className="input w-full mt-2"
                onChange={(e) => updateVectorSource('load_from_file', { file_path: e.target.value })}
              />
            )}
          </div>

          <button
            onClick={handleAddControl}
            disabled={!newControl.control_point}
            className="btn-primary w-full flex items-center justify-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>Add Control</span>
          </button>
        </div>
      </div>

      {/* Active controls */}
      <div className="flex-1 p-6 overflow-y-auto">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-secondary-300">
            Active Controls ({controls.activeControls.length})
          </h3>
          {controls.activeControls.length > 0 && (
            <div className="flex space-x-2">
              <button
                onClick={applyControls}
                disabled={controls.isApplying}
                className="btn-success flex items-center space-x-1 text-xs px-3 py-1"
              >
                <Play className="w-3 h-3" />
                <span>Apply</span>
              </button>
              <button
                onClick={clearControls}
                className="btn-error flex items-center space-x-1 text-xs px-3 py-1"
              >
                <Trash2 className="w-3 h-3" />
                <span>Clear</span>
              </button>
            </div>
          )}
        </div>

        {controls.activeControls.length === 0 ? (
          <div className="text-center py-8">
            <Target className="w-12 h-12 text-secondary-500 mx-auto mb-4" />
            <p className="text-secondary-400">
              No active controls. Add a control vector to begin.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {controls.activeControls.map((control, index) => (
              <div
                key={index}
                className="bg-secondary-700/50 rounded-lg p-4 border border-secondary-600"
              >
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <span className="text-sm font-medium text-secondary-200">
                      Layer {control.layer_idx}
                    </span>
                    <span className="text-xs text-secondary-400 ml-2">
                      Strength: {control.strength}
                    </span>
                  </div>
                  <button
                    onClick={() => removeControl(index)}
                    className="text-error-400 hover:text-error-300 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <div className="text-xs text-secondary-400 space-y-1">
                  <div>
                    Point: {control.control_point.replace(/_/g, ' ')}
                  </div>
                  <div>
                    Source: {control.vector_source.type.replace(/_/g, ' ')}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ControlPanel;
