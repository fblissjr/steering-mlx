// components/Sidebar.tsx
import React from 'react';
import { 
  Upload, 
  Sliders, 
  FlaskConical, 
  Search, 
  Zap,
  FileText
} from 'lucide-react';
import { useAppStore } from '../stores/appStore';
import ModelLoader from './panels/ModelLoader';
import ControlPanel from './panels/ControlPanel';
import VectorDerivation from './panels/VectorDerivation';
import FeatureAnalysis from './panels/FeatureAnalysis';
import GenerationPanel from './panels/GenerationPanel';

interface SidebarProps {
  collapsed: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ collapsed }) => {
  const { ui, setActiveTab } = useAppStore();

  const tabs = [
    {
      id: 'controls' as const,
      icon: Sliders,
      label: 'Controls',
      component: ControlPanel,
    },
    {
      id: 'derivation' as const,
      icon: FlaskConical,
      label: 'Derivation',
      component: VectorDerivation,
    },
    {
      id: 'analysis' as const,
      icon: Search,
      label: 'Analysis',
      component: FeatureAnalysis,
    },
    {
      id: 'generation' as const,
      icon: Zap,
      label: 'Generation',
      component: GenerationPanel,
    },
  ];

  const activeTabData = tabs.find(tab => tab.id === ui.activeTab);
  const ActiveComponent = activeTabData?.component;

  if (collapsed) {
    return (
      <div className="w-16 bg-secondary-800 border-r border-secondary-700 flex flex-col">
        {/* Collapsed tab icons */}
        <div className="flex flex-col p-2 space-y-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`p-3 rounded-lg transition-colors ${
                  ui.activeTab === tab.id 
                    ? 'bg-primary-600 text-white' 
                    : 'text-secondary-400 hover:bg-secondary-700 hover:text-secondary-200'
                }`}
                title={tab.label}
              >
                <Icon className="w-5 h-5" />
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="w-96 bg-secondary-800 border-r border-secondary-700 flex flex-col">
      {/* Model loader section - always visible */}
      <div className="border-b border-secondary-700">
        <ModelLoader />
      </div>

      {/* Tab navigation */}
      <div className="flex border-b border-secondary-700">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 text-sm font-medium transition-colors ${
                ui.activeTab === tab.id 
                  ? 'text-primary-400 border-b-2 border-primary-500 bg-secondary-700/50' 
                  : 'text-secondary-400 hover:text-secondary-200 hover:bg-secondary-700/30'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Active tab content */}
      <div className="flex-1 overflow-hidden">
        {ActiveComponent && <ActiveComponent />}
      </div>
    </div>
  );
};

export default Sidebar;
