# MLX Control Vector Laboratory - Frontend

React + TypeScript frontend for the MLX Control Vector Laboratory.

## Features

- **Interactive Model Visualization**: Visual representation of model architecture using React Flow
- **Multi-Model Support**: Supports Gemma 3, Llama, and Mixtral architectures  
- **Control Vector Management**: Add, configure, and apply control vectors at specific layers
- **Vector Derivation**: Derive control vectors from positive/negative prompt pairs
- **Feature Analysis**: Analyze where features are encoded within the model
- **Real-time Generation**: Generate text with active control vectors
- **Modern UI**: Dark theme with Tailwind CSS and smooth animations

## Tech Stack

- **React 18** with TypeScript
- **Vite** for development and building
- **Tailwind CSS v3** for styling
- **React Flow** for interactive model visualization
- **Zustand** for state management
- **Axios** for API communication
- **React Hot Toast** for notifications
- **Lucide React** for icons

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Project Structure

```
src/
├── components/           # React components
│   ├── panels/          # Sidebar panels (controls, derivation, etc.)
│   ├── visualization/   # Model visualization components
│   └── ...              # Shared components
├── stores/              # Zustand state management
├── services/            # API service layer
└── ...                  # App entry point, styles, etc.
```

## API Integration

The frontend communicates with the FastAPI backend running on port 8008. API calls are proxied through Vite during development.

## Key Components

- **ModelVisualization**: Interactive React Flow diagram of model architecture
- **LayerNode**: Interactive decoder layer with expandable control points
- **ControlPanel**: Manage and apply control vectors
- **VectorDerivation**: Derive control vectors from prompt pairs
- **FeatureAnalysis**: Analyze feature representations across layers
- **GenerationPanel**: Generate text with current model state

## State Management

Uses Zustand for simple, TypeScript-friendly state management:

- **Model State**: Loaded model info and status
- **Control State**: Active control vectors and derived vectors
- **Generation State**: Text generation results
- **Analysis State**: Feature analysis results  
- **UI State**: Selected layers, control points, and UI preferences
