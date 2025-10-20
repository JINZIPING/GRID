# Semantic ID Standalone Module

This is a standalone implementation of semantic ID generation using Kmeans clustering, extracted from the GRID project. It removes all Lightning dependencies and provides a pure PyTorch implementation.

## Features

- **Pure PyTorch**: No Lightning dependencies
- **Kmeans Clustering**: Mini-batch K-means with residual quantization
- **Multi-hierarchy**: Support for multiple quantization layers
- **Easy Integration**: Self-contained module for easy porting to other projects

## Files

- `semantic_id_module.py`: Main implementation with all classes
- `train_example.py`: Training example with dummy data
- `inference_example.py`: Inference example
- `test_complete.py`: Complete test suite
- `requirements.txt`: Required dependencies

## Quick Start

1. Install dependencies:
```bash
pip install torch numpy
```

2. Run the complete test:
```bash
python test_complete.py
```

3. Use in your project:
```python
from semantic_id_module import ResidualQuantization, MiniBatchKMeans, SquaredEuclideanDistance, KMeansPlusPlusInitInitializer

# Create model
model = ResidualQuantization(
    n_layers=3,
    quantization_layer=MiniBatchKMeans(...),
    # ... other parameters
)

# Train
model.train()
# ... training loop

# Inference
model.eval()
predictions = model.predict_step(embeddings)
```

## Model Architecture

The module implements residual quantization with multiple K-means layers:

1. **Input**: Item embeddings (e.g., 2048-dimensional)
2. **Quantization**: Multiple K-means layers (e.g., 3 hierarchies)
3. **Output**: Semantic IDs (e.g., 3-dimensional cluster assignments)

## Parameters

- `n_layers`: Number of quantization hierarchies
- `codebook_width`: Number of clusters per hierarchy (e.g., 256)
- `embedding_dim`: Dimension of input embeddings (e.g., 2048)
- `normalize_residuals`: Whether to normalize residuals between layers
- `train_layer_wise`: Whether to train layers sequentially

## Test Results

The test shows successful training and inference:
- ✅ Training: 3 epochs with decreasing loss
- ✅ Inference: Generated 1000 semantic IDs
- ✅ Quality: 5 unique combinations across 3 hierarchies
- ✅ Performance: ~200 similar items per semantic ID

## Integration

This module can be easily integrated into any PyTorch project:

1. Copy `semantic_id_module.py` to your project
2. Import the required classes
3. Create and train the model
4. Use for inference

No external dependencies beyond PyTorch and NumPy.
