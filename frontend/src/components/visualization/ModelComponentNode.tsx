// components/visualization/ModelComponentNode.tsx
import React from 'react';
import { Handle, Position } from 'reactflow';

interface ModelComponentNodeProps {
  data: {
    label: string;
    type: string;
    info?: any;
    onSelect: () => void;
  };
  selected?: boolean;
}

const ModelComponentNode: React.FC<ModelComponentNodeProps> = ({ data, selected }) => {
  const getNodeStyle = (type: string) => {
    switch (type) {
      case 'model':
        return 'bg-gradient-to-r from-primary-600 to-primary-700 text-white border-primary-500';
      case 'embedding':
        return 'bg-gradient-to-r from-purple-600 to-purple-700 text-white border-purple-500';
      case 'norm':
        return 'bg-gradient-to-r from-green-600 to-green-700 text-white border-green-500';
      case 'lm_head':
        return 'bg-gradient-to-r from-orange-600 to-orange-700 text-white border-orange-500';
      case 'attention':
        return 'bg-gradient-to-r from-blue-600 to-blue-700 text-white border-blue-500';
      case 'mlp':
        return 'bg-gradient-to-r from-red-600 to-red-700 text-white border-red-500';
      case 'linear':
        return 'bg-gradient-to-r from-secondary-600 to-secondary-700 text-white border-secondary-500';
      default:
        return 'bg-gradient-to-r from-secondary-600 to-secondary-700 text-white border-secondary-500';
    }
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'model':
        return '🧠';
      case 'embedding':
        return '📝';
      case 'norm':
        return '📏';
      case 'lm_head':
        return '🎯';
      case 'attention':
        return '👁️';
      case 'mlp':
        return '⚡';
      case 'linear':
        return '📊';
      default:
        return '⚙️';
    }
  };

  return (
    <div
      className={`
        px-4 py-3 rounded-lg border-2 cursor-pointer transition-all duration-200 min-w-[180px]
        ${getNodeStyle(data.type)}
        ${selected ? 'ring-2 ring-white ring-opacity-50 shadow-lg' : 'hover:shadow-md'}
      `}
      onClick={data.onSelect}
    >
      <Handle type="target" position={Position.Top} className="w-3 h-3" />
      
      <div className="flex items-center space-x-3">
        <div className="text-2xl">{getIcon(data.type)}</div>
        <div className="flex-1">
          <div className="font-semibold text-sm">{data.label}</div>
          <div className="text-xs opacity-80 capitalize">{data.type}</div>
        </div>
      </div>

      {/* Additional info for certain node types */}
      {data.info && (
        <div className="mt-2 text-xs opacity-75">
          {data.type === 'model' && (
            <div className="space-y-1">
              <div>{data.info.layers} layers</div>
              <div>{data.info.hidden_size}d hidden</div>
            </div>
          )}
          {data.type === 'embedding' && data.info.params && (
            <div>
              {data.info.params.vocab_size} vocab × {data.info.params.hidden_size}d
            </div>
          )}
          {data.type === 'linear' && data.info.params && (
            <div>
              → {data.info.params.out_features}
            </div>
          )}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
    </div>
  );
};

export default ModelComponentNode;
