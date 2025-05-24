// components/MainContent.tsx
import React from 'react';
import { useAppStore } from '../stores/appStore';
import ModelVisualization from './visualization/ModelVisualization';
import OutputPanel from './OutputPanel';

const MainContent: React.FC = () => {
  const { model } = useAppStore();

  return (
    <div className="flex-1 flex flex-col">
      {/* Main visualization area */}
      <div className="flex-1 relative">
        {model.isLoaded && model.modelInfo ? (
          <ModelVisualization />
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-center space-y-4">
              <div className="w-24 h-24 mx-auto bg-secondary-700 rounded-2xl flex items-center justify-center">
                <div className="w-12 h-12 border-4 border-secondary-500 border-dashed rounded-xl"></div>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-secondary-300 mb-2">
                  No Model Loaded
                </h3>
                <p className="text-secondary-400 max-w-md">
                  Load a model from the sidebar to begin exploring its architecture
                  and experimenting with control vectors.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Output panel */}
      <OutputPanel />
    </div>
  );
};

export default MainContent;
