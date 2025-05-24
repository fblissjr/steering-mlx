// components/visualization/LayerNode.tsx
import React, { useState } from 'react';
import { Handle, Position } from 'reactflow';
import { ChevronDown, ChevronRight, Target, Zap } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';

interface LayerNodeProps {
  data: {
    label: string;
    layerIdx: number;
    type: string;
    modelType: string;
    controlPoints: string[];
    isSelected: boolean;
    onSelect: (layerIdx: number) => void;
    onControlPointSelect: (point: string) => void;
  };
  selected?: boolean;
}

const LayerNode: React.FC<LayerNodeProps> = ({ data, selected }) => {
  const { controls, ui } = useAppStore();
  const [expanded, setExpanded] = useState(data.isSelected || selected);

  // Check if this layer has active controls
  const hasActiveControls = controls.activeControls.some(
    control => control.layer_idx === data.layerIdx
  );

  // Get active control points for this layer
  const activeControlPoints = controls.activeControls
    .filter(control => control.layer_idx === data.layerIdx)
    .map(control => control.control_point);

  React.useEffect(() => {
    if (data.isSelected) {
      setExpanded(true);
    }
  }, [data.isSelected]);

  const handleLayerClick = () => {
    data.onSelect(data.layerIdx);
    setExpanded(!expanded);
  };

  const handleControlPointClick = (point: string, e: React.MouseEvent) => {
    e.stopPropagation();
    data.onControlPointSelect(point);
  };

  const getLayerColor = () => {
    if (hasActiveControls) return 'border-success-500 bg-success-900/20';
    if (data.isSelected) return 'border-primary-500 bg-primary-900/20';
    return 'border-secondary-600 bg-secondary-800';
  };

  const getControlPointStatus = (point: string) => {
    if (activeControlPoints.includes(point)) return 'active';
    if (ui.selectedControlPoint === point && ui.selectedLayer === data.layerIdx) return 'selected';
    return 'inactive';
  };

  const getControlPointStyle = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-success-600 text-white border-success-500 ring-2 ring-success-400 ring-opacity-50';
      case 'selected':
        return 'bg-primary-600 text-white border-primary-500 ring-2 ring-primary-400 ring-opacity-50';
      default:
        return 'bg-secondary-700 text-secondary-300 border-secondary-600 hover:bg-secondary-600 hover:text-secondary-200';
    }
  };

  return (
    <div
      className={`
        rounded-lg border-2 cursor-pointer transition-all duration-200 min-w-[280px] max-w-[400px]
        ${getLayerColor()}
        ${selected ? 'ring-2 ring-white ring-opacity-30' : ''}
        hover:shadow-lg
      `}
    >
      <Handle type="target" position={Position.Top} className="w-3 h-3" />
      
      {/* Layer header */}
      <div
        className="px-4 py-3 flex items-center justify-between"
        onClick={handleLayerClick}
      >
        <div className="flex items-center space-x-3">
          <div className="text-lg">🧱</div>
          <div>
            <div className="font-semibold text-secondary-100">{data.label}</div>
            <div className="text-xs text-secondary-400 capitalize">
              {data.type.replace(/([A-Z])/g, ' $1').trim()}
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {hasActiveControls && (
            <div className="flex items-center space-x-1 px-2 py-1 bg-success-600 rounded-full">
              <Zap className="w-3 h-3 text-white" />
              <span className="text-xs text-white font-medium">
                {activeControlPoints.length}
              </span>
            </div>
          )}
          
          <button
            onClick={(e) => {
              e.stopPropagation();
              setExpanded(!expanded);
            }}
            className="text-secondary-400 hover:text-secondary-200 transition-colors"
          >
            {expanded ? (
              <ChevronDown className="w-4 h-4" />
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Expanded control points */}
      {expanded && data.controlPoints.length > 0 && (
        <div className="border-t border-secondary-700 p-4">
          <div className="mb-3 flex items-center space-x-2">
            <Target className="w-4 h-4 text-secondary-400" />
            <span className="text-sm font-medium text-secondary-300">
              Control Points
            </span>
          </div>
          
          <div className="grid grid-cols-1 gap-2">
            {data.controlPoints.map((point) => {
              const status = getControlPointStatus(point);
              return (
                <button
                  key={point}
                  onClick={(e) => handleControlPointClick(point, e)}
                  className={`
                    px-3 py-2 rounded-lg border text-xs font-medium transition-all duration-200 text-left
                    ${getControlPointStyle(status)}
                  `}
                  title={`Select ${point.replace(/_/g, ' ')} for control vector application`}
                >
                  <div className="flex items-center justify-between">
                    <span>
                      {point.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </span>
                    {status === 'active' && (
                      <Zap className="w-3 h-3" />
                    )}
                    {status === 'selected' && (
                      <Target className="w-3 h-3" />
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
    </div>
  );
};

export default LayerNode;
