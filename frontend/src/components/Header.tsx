// components/Header.tsx
import React from 'react';
import { 
  Cpu, 
  Zap, 
  Settings, 
  HelpCircle, 
  Activity,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { useAppStore } from '../stores/appStore';

const Header: React.FC = () => {
  const { model, ui, setSidebarCollapsed } = useAppStore();

  return (
    <header className="h-16 bg-secondary-800 border-b border-secondary-700 flex items-center justify-between px-6">
      {/* Left section */}
      <div className="flex items-center space-x-4">
        {/* Sidebar toggle */}
        <button
          onClick={() => setSidebarCollapsed(!ui.sidebarCollapsed)}
          className="p-2 hover:bg-secondary-700 rounded-lg transition-colors"
          title={ui.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {ui.sidebarCollapsed ? (
            <ChevronRight className="w-5 h-5 text-secondary-400" />
          ) : (
            <ChevronLeft className="w-5 h-5 text-secondary-400" />
          )}
        </button>

        {/* Logo and title */}
        <div className="flex items-center space-x-3">
          <div className="relative">
            <Cpu className="w-8 h-8 text-primary-400" />
            <Zap className="w-4 h-4 text-warning-400 absolute -top-1 -right-1" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gradient">
              MLX Control Vector Laboratory
            </h1>
            <p className="text-xs text-secondary-400">
              Interactive LLM Steering Platform
            </p>
          </div>
        </div>
      </div>

      {/* Center section - Model status */}
      <div className="flex items-center space-x-4">
        {model.isLoaded && model.modelInfo ? (
          <div className="flex items-center space-x-3 px-4 py-2 bg-secondary-700 rounded-lg">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-success-400 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-secondary-200">
                {model.modelInfo.model_type.toUpperCase()}
              </span>
            </div>
            <div className="text-xs text-secondary-400">
              {model.modelInfo.num_layers} layers • {model.modelInfo.hidden_size}d
            </div>
            {model.modelInfo.is_controlled && (
              <div className="px-2 py-1 bg-primary-600 text-xs rounded text-white font-medium">
                CONTROLLED
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center space-x-2 px-4 py-2 bg-secondary-700 rounded-lg text-secondary-400">
            <div className="w-2 h-2 bg-secondary-500 rounded-full"></div>
            <span className="text-sm">No model loaded</span>
          </div>
        )}
      </div>

      {/* Right section */}
      <div className="flex items-center space-x-2">
        {/* Trace mode toggle */}
        <button
          onClick={() => useAppStore.getState().setTraceMode(!ui.traceMode)}
          className={`p-2 rounded-lg transition-colors ${
            ui.traceMode 
              ? 'bg-primary-600 text-white' 
              : 'hover:bg-secondary-700 text-secondary-400'
          }`}
          title={ui.traceMode ? 'Disable trace mode' : 'Enable trace mode'}
        >
          <Activity className="w-5 h-5" />
        </button>

        {/* Settings */}
        <button
          className="p-2 hover:bg-secondary-700 rounded-lg transition-colors"
          title="Settings"
        >
          <Settings className="w-5 h-5 text-secondary-400" />
        </button>

        {/* Help */}
        <button
          className="p-2 hover:bg-secondary-700 rounded-lg transition-colors"
          title="Help"
        >
          <HelpCircle className="w-5 h-5 text-secondary-400" />
        </button>
      </div>
    </header>
  );
};

export default Header;
