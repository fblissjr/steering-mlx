// components/visualization/ModelVisualization.tsx
import React, { useCallback, useMemo, useState } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  MiniMap,
  ReactFlowProvider,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { useAppStore } from '../../stores/appStore';
import ModelComponentNode from './ModelComponentNode';
import LayerNode from './LayerNode';
import ControlPointNode from './ControlPointNode';

const nodeTypes = {
  modelComponent: ModelComponentNode,
  layer: LayerNode,
  controlPoint: ControlPointNode,
};

const ModelVisualization: React.FC = () => {
  const { model, ui } = useAppStore();
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  // Generate nodes and edges from architecture info
  const { nodes: initialNodes, edges: initialEdges } = useMemo(() => {
    if (!model.modelInfo?.architecture_info) {
      return { nodes: [], edges: [] };
    }

    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const archInfo = model.modelInfo.architecture_info;

    let yOffset = 0;
    const LAYER_HEIGHT = 120;
    const COMPONENT_WIDTH = 200;

    // Root model node
    nodes.push({
      id: 'root',
      type: 'modelComponent',
      position: { x: 400, y: yOffset },
      data: {
        label: archInfo.model_name || 'Model',
        type: 'model',
        info: {
          type: archInfo.model_type,
          layers: archInfo.num_layers,
          hidden_size: archInfo.hidden_size,
          vocab_size: archInfo.vocab_size,
        },
        onSelect: () => {},
      },
    });

    yOffset += 100;

    // Embedding layer
    if (archInfo.layer_structure?.embed_tokens) {
      nodes.push({
        id: 'embed_tokens',
        type: 'modelComponent',
        position: { x: 400, y: yOffset },
        data: {
          label: 'Token Embeddings',
          type: 'embedding',
          info: archInfo.layer_structure.embed_tokens,
          onSelect: () => {},
        },
      });

      edges.push({
        id: 'root-embed',
        source: 'root',
        target: 'embed_tokens',
        type: 'smoothstep',
      });

      yOffset += 80;
    }

    // Decoder layers
    if (archInfo.layer_structure?.layers && archInfo.num_layers > 0) {
      const layerStructure = archInfo.layer_structure.layers.item_structure;
      
      for (let i = 0; i < archInfo.num_layers; i++) {
        const layerId = `layer_${i}`;
        
        // Main layer node
        nodes.push({
          id: layerId,
          type: 'layer',
          position: { x: 200, y: yOffset },
          data: {
            label: `Layer ${i}`,
            layerIdx: i,
            type: layerStructure?.type || 'DecoderLayer',
            modelType: archInfo.model_type,
            controlPoints: layerStructure?.control_points || [],
            isSelected: ui.selectedLayer === i,
            onSelect: (layerIdx: number) => {
              useAppStore.getState().setSelectedLayer(layerIdx);
              setSelectedNode(layerId);
            },
            onControlPointSelect: (point: string) => {
              useAppStore.getState().setSelectedControlPoint(point);
              useAppStore.getState().setSelectedLayer(i);
            },
          },
        });

        // Sub-components for expanded view
        if (layerStructure?.sub_modules && ui.selectedLayer === i) {
          let subXOffset = 450;
          layerStructure.sub_modules.forEach((subModule: any, subIdx: number) => {
            const subId = `${layerId}_${subModule.name}`;
            nodes.push({
              id: subId,
              type: 'modelComponent',
              position: { x: subXOffset, y: yOffset + 20 },
              data: {
                label: subModule.name.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()),
                type: subModule.type?.toLowerCase() || 'component',
                info: subModule,
                onSelect: () => setSelectedNode(subId),
              },
            });

            edges.push({
              id: `${layerId}-${subId}`,
              source: layerId,
              target: subId,
              type: 'smoothstep',
            });

            subXOffset += 180;
          });
        }

        // Connect layers
        if (i === 0) {
          edges.push({
            id: `embed-${layerId}`,
            source: 'embed_tokens',
            target: layerId,
            type: 'smoothstep',
          });
        } else {
          edges.push({
            id: `layer_${i-1}-${layerId}`,
            source: `layer_${i-1}`,
            target: layerId,
            type: 'smoothstep',
          });
        }

        yOffset += LAYER_HEIGHT;
      }
    }

    // Final norm and LM head
    if (archInfo.layer_structure?.norm) {
      const normId = 'final_norm';
      nodes.push({
        id: normId,
        type: 'modelComponent',
        position: { x: 400, y: yOffset },
        data: {
          label: 'Final Norm',
          type: 'norm',
          info: archInfo.layer_structure.norm,
          onSelect: () => setSelectedNode(normId),
        },
      });

      if (archInfo.num_layers > 0) {
        edges.push({
          id: `layer_${archInfo.num_layers-1}-${normId}`,
          source: `layer_${archInfo.num_layers-1}`,
          target: normId,
          type: 'smoothstep',
        });
      }

      yOffset += 80;
    }

    if (archInfo.layer_structure?.lm_head) {
      const lmHeadId = 'lm_head';
      nodes.push({
        id: lmHeadId,
        type: 'modelComponent',
        position: { x: 400, y: yOffset },
        data: {
          label: 'Language Model Head',
          type: 'lm_head',
          info: archInfo.layer_structure.lm_head,
          onSelect: () => setSelectedNode(lmHeadId),
        },
      });

      edges.push({
        id: 'norm-lmhead',
        source: 'final_norm',
        target: lmHeadId,
        type: 'smoothstep',
      });
    }

    return { nodes, edges };
  }, [model.modelInfo?.architecture_info, ui.selectedLayer]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback((params: Connection) => {
    setEdges((eds) => addEdge(params, eds));
  }, [setEdges]);

  // Update nodes when selection changes
  React.useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.type === 'layer') {
          return {
            ...node,
            data: {
              ...node.data,
              isSelected: ui.selectedLayer === node.data.layerIdx,
            },
          };
        }
        return node;
      })
    );
  }, [ui.selectedLayer, setNodes]);

  if (!model.modelInfo?.architecture_info) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-secondary-600 border-dashed rounded-xl mx-auto mb-4"></div>
          <p className="text-secondary-400">No model architecture to visualize</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full w-full">
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{
            padding: 0.2,
            includeHiddenNodes: false,
          }}
          minZoom={0.1}
          maxZoom={2}
          defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
        >
          <Background 
            variant="dots" 
            gap={20} 
            size={1}
            color="#475569"
          />
          <Controls 
            className="bg-secondary-800 border-secondary-700"
            showInteractive={false}
          />
          <MiniMap 
            className="bg-secondary-800 border-secondary-700"
            nodeColor={(node) => {
              switch (node.type) {
                case 'layer': return '#3b82f6';
                case 'modelComponent': return '#64748b';
                case 'controlPoint': return '#22c55e';
                default: return '#64748b';
              }
            }}
            nodeStrokeWidth={2}
            pannable
            zoomable
          />
        </ReactFlow>
      </ReactFlowProvider>

      {/* Info panel for selected node */}
      {selectedNode && (
        <div className="absolute top-4 right-4 w-80 bg-secondary-800/95 backdrop-blur-sm rounded-xl border border-secondary-700 p-4 shadow-xl">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-secondary-100">
              Node Details
            </h3>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-secondary-400 hover:text-secondary-200 transition-colors"
            >
              ×
            </button>
          </div>
          
          <div className="space-y-2 text-sm">
            <div>
              <span className="text-secondary-400">ID:</span>
              <span className="text-secondary-200 ml-2">{selectedNode}</span>
            </div>
            {ui.selectedLayer !== null && (
              <div>
                <span className="text-secondary-400">Selected Layer:</span>
                <span className="text-primary-400 ml-2">{ui.selectedLayer}</span>
              </div>
            )}
            {ui.selectedControlPoint && (
              <div>
                <span className="text-secondary-400">Selected Control Point:</span>
                <span className="text-success-400 ml-2">{ui.selectedControlPoint.replace(/_/g, ' ')}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelVisualization;
