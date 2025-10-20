# Semantic ID Standalone Module

A self-contained implementation of semantic ID generation using K-means clustering and residual quantization, without external dependencies beyond PyTorch and NumPy.

## 📁 Project Structure

```
semantic_id_standalone_module/
├── __init__.py                 # Main module exports
├── requirements.txt            # Dependencies
├── src/                        # Core implementation
│   ├── __init__.py
│   └── semantic_id_module.py   # Main implementation
├── examples/                   # Usage examples
│   ├── train_example.py        # Training example
│   └── inference_example.py    # Inference example
├── tests/                      # Test suite
│   ├── __init__.py
│   └── test_complete.py        # Complete test suite
└── docs/                       # Documentation
    ├── README.md               # Detailed documentation
    └── SUMMARY.md              # Feature summary
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install torch numpy
```

### Basic Usage

```python
from semantic_id_standalone_module import (
    ResidualQuantization,
    MiniBatchKMeans,
    SquaredEuclideanDistance,
    KMeansPlusPlusInitInitializer
)

# Create model
quantization_layer = MiniBatchKMeans(
    n_clusters=256,
    n_features=2048,
    distance_function=SquaredEuclideanDistance(),
    initializer=KMeansPlusPlusInitInitializer(
        n_clusters=256,
        distance_function=SquaredEuclideanDistance(),
        initialize_on_cpu=False
    ),
    init_buffer_size=3072,
    optimizer=None,
)

model = ResidualQuantization(
    n_layers=3,
    quantization_layer=quantization_layer,
    init_buffer_size=3072,
    normalize_residuals=True,
    train_layer_wise=True,
    track_residuals=True,
    verbose=True,
)

# Training
model.train()
# ... training loop ...

# Inference
model.eval()
with torch.no_grad():
    predictions = model.predict_step(embeddings)
```

## 📖 Examples

- **Training**: See `examples/train_example.py`
- **Inference**: See `examples/inference_example.py`
- **Complete Test**: See `tests/test_complete.py`

## 🔧 Configuration

The module uses the same parameters as the original implementation:

- `embedding_dim`: 2048 (model dimension)
- `num_hierarchies`: 3 (number of quantization layers)
- `codebook_width`: 256 (centroids per layer)
- `init_buffer_size`: 3072 (initialization buffer size)
- `optimizer`: SGD with lr=0.5

## 📚 Documentation

- [Detailed Documentation](docs/README.md)
- [Feature Summary](docs/SUMMARY.md)

## ✅ Features

- ✅ Pure PyTorch implementation (no Lightning dependency)
- ✅ K-means++ initialization
- ✅ Residual quantization
- ✅ Mini-batch K-means clustering
- ✅ Configurable parameters
- ✅ GPU/CPU support
- ✅ Complete test suite
- ✅ Usage examples

## 🔄 Migration from Original

This module is a standalone version of the original Lightning-based implementation. Key differences:

- Removed Lightning dependencies
- Simplified device management
- Self-contained implementation
- Same algorithms and parameters

## 📝 License

Same as the original GRID project.
