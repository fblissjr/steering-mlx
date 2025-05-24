// components/panels/GenerationPanel.tsx
import React, { useState } from 'react';
import { Zap, Send, Copy, RotateCcw, Settings2 } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';

const GenerationPanel: React.FC = () => {
  const { model, generation, generate } = useAppStore();
  
  const [prompt, setPrompt] = useState('');
  const [maxTokens, setMaxTokens] = useState(100);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(1.0);
  const [useChatTemplate, setUseChatTemplate] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    await generate({
      prompt: prompt.trim(),
      max_tokens: maxTokens,
      temp: temperature,
      top_p: topP,
    });
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const loadExample = (example: string) => {
    setPrompt(example);
  };

  const examplePrompts = [
    "What is the capital of France?",
    "Write a short poem about the stars.",
    "Explain quantum computing in simple terms.",
    "Tell me a story about a friendly robot.",
    "What are the benefits of renewable energy?",
  ];

  if (!model.isLoaded || !model.modelInfo) {
    return (
      <div className="p-6 text-center">
        <Zap className="w-12 h-12 text-secondary-500 mx-auto mb-4" />
        <p className="text-secondary-400">
          Load a model to generate text
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-6 border-b border-secondary-700">
        <div className="flex items-center space-x-2 mb-4">
          <Zap className="w-5 h-5 text-primary-400" />
          <h2 className="text-lg font-semibold text-secondary-100">
            Text Generation
          </h2>
        </div>

        {/* Prompt input */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-secondary-300 mb-2">
              Prompt
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt here..."
              className="input w-full h-32 resize-none"
              disabled={generation.isGenerating}
            />
          </div>

          {/* Quick examples */}
          <div>
            <label className="block text-sm font-medium text-secondary-300 mb-2">
              Quick Examples
            </label>
            <div className="flex flex-wrap gap-2">
              {examplePrompts.map((example, index) => (
                <button
                  key={index}
                  onClick={() => loadExample(example)}
                  className="text-xs px-3 py-1 bg-secondary-700 hover:bg-secondary-600 rounded-full transition-colors text-secondary-300"
                  disabled={generation.isGenerating}
                >
                  {example.substring(0, 30)}...
                </button>
              ))}
            </div>
          </div>

          {/* Basic settings */}
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-sm font-medium text-secondary-300 mb-1">
                Max Tokens
              </label>
              <input
                type="number"
                min="1"
                max="2048"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value) || 100)}
                className="input w-full"
                disabled={generation.isGenerating}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-secondary-300 mb-1">
                Temperature
              </label>
              <input
                type="number"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value) || 0.7)}
                className="input w-full"
                disabled={generation.isGenerating}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-secondary-300 mb-1">
                Top-p
              </label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value) || 1.0)}
                className="input w-full"
                disabled={generation.isGenerating}
              />
            </div>
          </div>

          {/* Advanced settings toggle */}
          <div>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center space-x-2 text-sm text-secondary-400 hover:text-secondary-200 transition-colors"
            >
              <Settings2 className="w-4 h-4" />
              <span>{showAdvanced ? 'Hide' : 'Show'} Advanced Options</span>
            </button>

            {showAdvanced && (
              <div className="mt-3 space-y-3 p-3 bg-secondary-900/50 rounded-lg border border-secondary-700">
                <div className="flex items-center space-x-3">
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={useChatTemplate}
                      onChange={(e) => setUseChatTemplate(e.target.checked)}
                      className="rounded border-secondary-600 bg-secondary-700 text-primary-600 focus:ring-primary-500"
                      disabled={generation.isGenerating}
                    />
                    <span className="text-sm text-secondary-300">
                      Use chat template
                    </span>
                  </label>
                </div>

                {useChatTemplate && (
                  <div>
                    <label className="block text-sm font-medium text-secondary-300 mb-1">
                      System Prompt
                    </label>
                    <textarea
                      value={systemPrompt}
                      onChange={(e) => setSystemPrompt(e.target.value)}
                      placeholder="Enter system prompt..."
                      className="input w-full h-20 resize-none"
                      disabled={generation.isGenerating}
                    />
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Generate button */}
          <button
            onClick={handleGenerate}
            disabled={generation.isGenerating || !prompt.trim()}
            className="btn-primary w-full flex items-center justify-center space-x-2"
          >
            <Send className="w-4 h-4" />
            <span>{generation.isGenerating ? 'Generating...' : 'Generate'}</span>
          </button>
        </div>
      </div>

      {/* Generation output */}
      <div className="flex-1 p-6 overflow-y-auto">
        {generation.lastGeneration ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-secondary-300">
                Generated Text
              </h3>
              <div className="flex space-x-2">
                <button
                  onClick={() => copyToClipboard(generation.lastGeneration!.response)}
                  className="text-secondary-400 hover:text-secondary-200 transition-colors"
                  title="Copy to clipboard"
                >
                  <Copy className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setPrompt(generation.lastGeneration!.prompt)}
                  className="text-secondary-400 hover:text-secondary-200 transition-colors"
                  title="Reuse prompt"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Original prompt */}
            <div className="p-3 bg-secondary-700/30 rounded-lg border border-secondary-600">
              <div className="text-xs font-medium text-secondary-400 mb-2">
                Original Prompt:
              </div>
              <div className="text-sm text-secondary-300">
                {generation.lastGeneration.prompt}
              </div>
            </div>

            {/* Generated response */}
            <div className="p-4 bg-secondary-700/50 rounded-lg border border-secondary-600">
              <div className="text-sm text-secondary-100 whitespace-pre-wrap leading-relaxed">
                {generation.lastGeneration.response}
              </div>
            </div>

            {/* Metadata */}
            <div className="flex justify-between text-xs text-secondary-400">
              <span>
                Generated {new Date(generation.lastGeneration.timestamp).toLocaleTimeString()}
              </span>
              <span>
                {generation.lastGeneration.response.split(/\s+/).length} words
              </span>
            </div>
          </div>
        ) : (
          <div className="text-center py-12">
            <Zap className="w-12 h-12 text-secondary-500 mx-auto mb-4" />
            <p className="text-secondary-400">
              No generation yet. Enter a prompt and click generate to see results.
            </p>
          </div>
        )}

        {/* Error display */}
        {generation.error && (
          <div className="mt-4 p-3 bg-error-900/20 border border-error-700 rounded-lg">
            <div className="text-sm font-medium text-error-400 mb-1">
              Generation Error:
            </div>
            <div className="text-xs text-error-300">
              {generation.error}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default GenerationPanel;
