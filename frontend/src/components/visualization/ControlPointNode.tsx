// components/visualization/ControlPointNode.tsx
import React from 'react';
import { Handle, Position } from 'reactflow';
import { Target, Zap, Eye } from 'lucide-react';

interface ControlPointNodeProps {
  data: {
    label: string;
    controlPoint: string;
    layerIdx: number;
    isActive: boolean;
    isSelected: boolean;
    onSelect: () => void;
  };
  selected?: boolean;
}

const ControlPointNode: React.FC<ControlPointNodeProps> = ({ data, selected }) => {
  const getNodeStyle = () => {
    if (data.isActive) {
      return 'bg-gradient-to-r from-success-600 to-success-700 text-white border-success-500 ring-2 ring-success-400 ring-opacity-50';
    }
    if (data.isSelected) {
      return 'bg-gradient-to-r from-primary-600 to-primary-700 text-white border-primary-500 ring-2 ring-primary-400 ring-opacity-50';
    }
    return 'bg-gradient-to-r from-secondary-600 to-secondary-700 text-white border-secondary-500 hover:from-secondary-500 hover:to-secondary-600';
  };

  const getIcon = () => {
    if (data.isActive) return <Zap className="w-4 h-4" />;
    if (data.isSelected) return <Target className="w-4 h-4" />;
    return <Eye className="w-4 h-4" />;
  };

  return (
    <div
      className={`
        px-3 py-2 rounded-lg border-2 cursor-pointer transition-all duration-200 min-w-[160px]
        ${getNodeStyle()}
        ${selected ? 'ring-2 ring-white ring-opacity-30' : ''}
        hover:shadow-md
      `}
      onClick={data.onSelect}
    >
      <Handle type="target" position={Position.Top} className="w-2 h-2" />
      
      <div className="flex items-center space-x-2">
        {getIcon()}
        <div className="flex-1">
          <div className="font-medium text-sm">{data.label}</div>
          <div className="text-xs opacity-75">
            Layer {data.layerIdx}
          </div>
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} className="w-2 h-2" />
    </div>
  );
};

export default ControlPointNode;
