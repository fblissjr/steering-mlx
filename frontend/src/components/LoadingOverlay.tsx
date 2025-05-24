// components/LoadingOverlay.tsx
import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingOverlayProps {
  message?: string;
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ 
  message = 'Loading...' 
}) => {
  return (
    <div className="fixed inset-0 bg-secondary-900/80 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-secondary-800 rounded-xl p-8 shadow-2xl border border-secondary-700 max-w-sm w-full mx-4">
        <div className="flex flex-col items-center space-y-4">
          <Loader2 className="w-12 h-12 text-primary-400 animate-spin" />
          <div className="text-center">
            <h3 className="text-lg font-semibold text-secondary-100 mb-2">
              {message}
            </h3>
            <p className="text-sm text-secondary-400">
              This may take a few moments...
            </p>
          </div>
          <div className="w-full bg-secondary-700 rounded-full h-2 overflow-hidden">
            <div className="h-full bg-gradient-to-r from-primary-500 to-purple-500 rounded-full animate-pulse"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingOverlay;
