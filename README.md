# MLX Control Vector Laboratory

**Interactive web-based platform for LLM control vector experimentation on Apple Silicon**

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-green)
![License](https://img.shields.io/badge/license-MIT-blue)

An advanced research tool for exploring and manipulating Large Language Model behavior through control vectors, built specifically for Apple's MLX framework.

## ✨ Features

### 🎯 **Interactive Model Visualization**
- Visual representation of model architecture using React Flow
- Click-to-target layers and control points for precise intervention
- Real-time trace animation showing model processing flow
- Support for multiple architectures (Gemma 3, Llama, Mixtral)

### ⚡ **Control Vector Operations**
- **Derive**: Extract control vectors from positive/negative prompt pairs
- **Apply**: Inject control vectors at specific model layers and activation points
- **Analyze**: Discover where semantic features are encoded in the model
- **Generate**: Test controlled model behavior with interactive text generation

### 🧠 **Multi-Model Architecture Support**
- **Gemma 3**: Full control support with sliding window attention
- **Llama**: Standard transformer architecture (framework ready)
- **Mixtral**: Mixture of Experts support (framework ready)

### 🛠️ **Advanced Experimentation**
- Six control points per decoder layer for granular intervention
- Quantization support (MLX native, AWQ, DWQ)
- Experiment configuration via JSON for reproducible research
- Feature analysis with multiple differentiation metrics

## 🚀 Quick Start

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.9+**
- **Node.js 18+**
- **MLX** and **MLX-LM** compatible environment

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd steering-mlx
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd ../frontend
   npm install
   ```

### Running the Application

1. **Start the Backend API**
   ```bash
   cd backend
   python main_api.py
   ```
   The API will be available at `http://127.0.0.1:8008`

2. **Start the Frontend** (in a new terminal)
   ```bash
   cd frontend
   npm run dev
   ```
   The web interface will be available at `http://localhost:3000`

3. **Test the Connection**
   ```bash
   cd ..
   python test_backend.py
   ```

## 📖 Usage Guide

### 1. Load a Model

**Popular Models to Try:**
- `mlx-community/gemma-2b-it-4bit` (Quick testing)
- `mlx-community/gemma-7b-it-4bit` (Balanced performance)
- `mlx-community/Llama-3.2-3B-Instruct-4bit` (Standard transformer)

### 2. Explore the Architecture

- Navigate the interactive model diagram
- Click on layers to explore their structure
- Select control points for intervention

### 3. Apply Control Vectors

**Quick Control:**
- Select a layer and control point
- Choose vector source (random, derived, or file)
- Set strength and apply

**Derive Custom Vectors:**
- Provide positive example prompts
- Provide negative example prompts  
- Derive and save the control vector

### 4. Analyze Features

- Define feature-positive and feature-negative prompts
- Run analysis across layers and control points
- Identify where semantic features are encoded

### 5. Generate and Test

- Enter prompts to test controlled behavior
- Compare with and without active controls
- Export results for further analysis

## 🏗️ Architecture

### Backend (`/backend`)
- **FastAPI** REST API for model operations
- **MLX-LM** integration for model loading and inference
- Multi-model controlled layer implementations
- Vector derivation and feature analysis utilities

### Frontend (`/frontend`)  
- **React + TypeScript** SPA with modern UI
- **React Flow** for interactive model visualization
- **Zustand** for state management
- **Tailwind CSS** for styling

### Key Components

```
steering-mlx/
├── backend/
│   ├── main_api.py              # FastAPI server
│   ├── control_core.py          # Multi-model control framework
│   ├── controlled_gemma3.py     # Gemma 3 implementation
│   ├── controlled_llama.py      # Llama implementation (placeholder)
│   ├── controlled_mixtral.py    # Mixtral implementation (placeholder)
│   ├── control_utils.py         # Vector operations
│   ├── feature_analyzer.py      # Feature analysis
│   └── model_loader.py          # Unified model loading
└── frontend/
    ├── src/components/
    │   ├── panels/              # Control panels
    │   └── visualization/       # Model visualization
    └── src/stores/              # State management
```

## 🔬 Advanced Usage

### Control Points Available

Each decoder layer provides six intervention points:
- `pre_attention_layernorm_input`
- `attention_output`
- `post_attention_residual`  
- `pre_mlp_layernorm_input`
- `mlp_output`
- `post_mlp_residual`

### Experiment Configuration

Create JSON experiment files:

```json
{
  "experiments": [{
    "name": "Encourage Conciseness",
    "description": "Make responses more concise",
    "controls": [{
      "layer_idx": 30,
      "control_point": "mlp_output",
      "strength": 1.5,
      "vector_source": {
        "type": "derive",
        "positive_prompts_raw": ["Be brief.", "Short answer:"],
        "negative_prompts_raw": ["Explain in detail...", "Write extensively..."]
      }
    }],
    "test_prompts": ["What is quantum computing?"]
  }]
}
```

### API Integration

The frontend communicates with the backend via REST API:

- `POST /load_model` - Load and wrap model with control capabilities
- `POST /apply_controls` - Apply control vector configurations  
- `POST /derive_vector` - Derive control vectors from prompts
- `POST /analyze_feature` - Run feature analysis
- `POST /generate` - Generate text with current controls

## 🛠️ Development

### Extending Model Support

To add support for new model architectures:

1. Create `controlled_<model>.py` in backend
2. Extend `ControlledLayerBase` for the model's decoder layers
3. Add model info to `MODEL_ARCHITECTURES` in `control_core.py`
4. Update frontend visualization for model-specific features

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request

## 📚 Research Applications

This tool enables research in:

- **Mechanistic Interpretability**: Understanding how models process information
- **AI Safety**: Studying and mitigating harmful model behaviors
- **Controlled Generation**: Steering model outputs for specific applications  
- **Feature Analysis**: Discovering how concepts are encoded in neural networks
- **Model Editing**: Precise modification of model behavior

## 🔧 Troubleshooting

### Common Issues

**Model Loading Fails:**
- Ensure MLX and MLX-LM are properly installed
- Check model path/HuggingFace ID is correct
- Verify sufficient memory for model size

**API Connection Issues:**
- Ensure backend is running on port 8008
- Check firewall settings
- Verify frontend proxy configuration

**Memory Issues:**
- Use quantized models (4-bit recommended)
- Close other applications to free memory
- Consider smaller models for testing

### Performance Tips

- Use 4-bit quantized models for faster loading
- Enable only necessary control points for analysis
- Use specific layer ranges rather than analyzing all layers

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on Apple's [MLX](https://github.com/ml-explore/mlx) framework
- Inspired by research in mechanistic interpretability and AI safety
- Frontend visualization powered by [React Flow](https://reactflow.dev/)

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in `/ref-docs`
- Review example experiments in `/ref-code`

---

**Happy experimenting! 🚀**
