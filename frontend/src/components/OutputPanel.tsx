// components/OutputPanel.tsx
import React, { useState } from 'react';
import { 
  FileText, 
  Activity, 
  BarChart3, 
  Terminal, 
  Copy, 
  Download,
  ChevronUp,
  ChevronDown 
} from 'lucide-react';
import { useAppStore } from '../stores/appStore';

const OutputPanel: React.FC = () => {
  const { generation, analysis } = useAppStore();
  const [activeTab, setActiveTab] = useState<'generation' | 'analysis' | 'trace' | 'logs'>('generation');
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [logs, setLogs] = useState<Array<{ timestamp: number; level: string; message: string }>>([
    { timestamp: Date.now(), level: 'info', message: 'MLX Control Vector Laboratory initialized' },
    { timestamp: Date.now() + 1000, level: 'info', message: 'Ready for model loading and experimentation' },
  ]);

  const tabs = [
    {
      id: 'generation' as const,
      icon: FileText,
      label: 'Generation',
      badge: generation.lastGeneration ? '1' : null,
    },
    {
      id: 'analysis' as const,
      icon: BarChart3,
      label: 'Analysis',
      badge: analysis.lastAnalysis ? String(analysis.lastAnalysis.results.length) : null,
    },
    {
      id: 'trace' as const,
      icon: Activity,
      label: 'Trace',
      badge: null,
    },
    {
      id: 'logs' as const,
      icon: Terminal,
      label: 'Logs',
      badge: String(logs.length),
    },
  ];

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const exportData = (data: any, filename: string) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (isCollapsed) {
    return (
      <div className="h-12 bg-secondary-800 border-t border-secondary-700 flex items-center justify-between px-6">
        <div className="flex items-center space-x-4">
          <div className="flex space-x-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <div key={tab.id} className="flex items-center space-x-1 text-secondary-400">
                  <Icon className="w-4 h-4" />
                  {tab.badge && (
                    <span className="px-2 py-1 bg-primary-600 text-xs rounded text-white">
                      {tab.badge}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
        <button
          onClick={() => setIsCollapsed(false)}
          className="text-secondary-400 hover:text-secondary-200 transition-colors"
        >
          <ChevronUp className="w-5 h-5" />
        </button>
      </div>
    );
  }

  return (
    <div className="h-80 bg-secondary-800 border-t border-secondary-700 flex flex-col">
      {/* Tab header */}
      <div className="flex items-center justify-between border-b border-secondary-700">
        <div className="flex">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-4 py-3 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'text-primary-400 border-b-2 border-primary-500 bg-secondary-700/50'
                    : 'text-secondary-400 hover:text-secondary-200 hover:bg-secondary-700/30'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
                {tab.badge && (
                  <span className="px-2 py-1 bg-primary-600 text-xs rounded text-white">
                    {tab.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
        <button
          onClick={() => setIsCollapsed(true)}
          className="p-2 text-secondary-400 hover:text-secondary-200 transition-colors mr-2"
        >
          <ChevronDown className="w-5 h-5" />
        </button>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'generation' && (
          <div className="p-4 h-full overflow-y-auto">
            {generation.lastGeneration ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium text-secondary-300">
                    Last Generation Result
                  </h3>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => copyToClipboard(generation.lastGeneration!.response)}
                      className="text-secondary-400 hover:text-secondary-200 transition-colors"
                      title="Copy response"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => exportData(generation.lastGeneration, 'generation_result.json')}
                      className="text-secondary-400 hover:text-secondary-200 transition-colors"
                      title="Export result"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs font-medium text-secondary-400 mb-2">Prompt:</div>
                    <div className="p-3 bg-secondary-700/30 rounded-lg text-sm text-secondary-300 max-h-32 overflow-y-auto">
                      {generation.lastGeneration.prompt}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs font-medium text-secondary-400 mb-2">Response:</div>
                    <div className="p-3 bg-secondary-700/50 rounded-lg text-sm text-secondary-100 max-h-32 overflow-y-auto whitespace-pre-wrap">
                      {generation.lastGeneration.response}
                    </div>
                  </div>
                </div>

                <div className="text-xs text-secondary-400 flex justify-between">
                  <span>Generated: {new Date(generation.lastGeneration.timestamp).toLocaleString()}</span>
                  <span>{generation.lastGeneration.response.split(/\s+/).length} words</span>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <FileText className="w-12 h-12 text-secondary-500 mx-auto mb-4" />
                <p className="text-secondary-400">No generation results yet</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="p-4 h-full overflow-y-auto">
            {analysis.lastAnalysis ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-medium text-secondary-300">
                      Feature Analysis Results
                    </h3>
                    <p className="text-xs text-secondary-400">
                      {analysis.lastAnalysis.model_type.toUpperCase()} • {analysis.lastAnalysis.results.length} locations analyzed
                    </p>
                  </div>
                  <button
                    onClick={() => exportData(analysis.lastAnalysis, 'analysis_results.json')}
                    className="text-secondary-400 hover:text-secondary-200 transition-colors"
                    title="Export results"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                </div>

                <div className="space-y-2">
                  <div className="grid grid-cols-4 gap-4 text-xs font-medium text-secondary-400 border-b border-secondary-700 pb-2">
                    <div>Rank</div>
                    <div>Layer</div>
                    <div>Control Point</div>
                    <div>Score</div>
                  </div>
                  {analysis.lastAnalysis.results.slice(0, 20).map((result, index) => (
                    <div
                      key={index}
                      className="grid grid-cols-4 gap-4 text-xs py-2 hover:bg-secondary-700/30 rounded transition-colors"
                    >
                      <div className="text-secondary-300">#{index + 1}</div>
                      <div className="text-secondary-200">L{result.layer_idx}</div>
                      <div className="text-secondary-300 truncate">
                        {result.control_point.replace(/_/g, ' ')}
                      </div>
                      <div className="text-primary-400 font-mono">
                        {result.differentiation_score.toFixed(4)}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="text-xs text-secondary-400">
                  Analysis completed: {new Date(analysis.lastAnalysis.timestamp).toLocaleString()}
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <BarChart3 className="w-12 h-12 text-secondary-500 mx-auto mb-4" />
                <p className="text-secondary-400">No analysis results yet</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'trace' && (
          <div className="p-4 h-full overflow-y-auto">
            <div className="text-center py-12">
              <Activity className="w-12 h-12 text-secondary-500 mx-auto mb-4" />
              <p className="text-secondary-400">
                Real-time model execution tracing will appear here
              </p>
              <p className="text-xs text-secondary-500 mt-2">
                Enable trace mode in the header to see live model processing
              </p>
            </div>
          </div>
        )}

        {activeTab === 'logs' && (
          <div className="p-4 h-full overflow-y-auto">
            <div className="space-y-2 font-mono text-xs">
              {logs.map((log, index) => (
                <div
                  key={index}
                  className={`flex items-center space-x-3 p-2 rounded ${
                    log.level === 'error' ? 'bg-error-900/20 text-error-300' :
                    log.level === 'warning' ? 'bg-warning-900/20 text-warning-300' :
                    'text-secondary-400'
                  }`}
                >
                  <span className="text-secondary-500">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    log.level === 'error' ? 'bg-error-600 text-white' :
                    log.level === 'warning' ? 'bg-warning-600 text-white' :
                    'bg-secondary-600 text-secondary-200'
                  }`}>
                    {log.level.toUpperCase()}
                  </span>
                  <span className="flex-1">{log.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default OutputPanel;
