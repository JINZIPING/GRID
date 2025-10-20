#!/usr/bin/env python3
"""
Simple inference example for semantic ID module.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.semantic_id_module import (
    ResidualQuantization, 
    MiniBatchKMeans, 
    SquaredEuclideanDistance, 
    KMeansPlusPlusInitInitializer
)


def main():
    """Simple inference example."""
    print("Semantic ID Inference Example")
    print("=" * 40)
    
    # Parameters (matching original config)
    embedding_dim = 2048
    num_hierarchies = 3
    codebook_width = 256
    init_buffer_size = 3072  # Match original config
    
    # Create model (same as training)
    print("Creating model...")
    quantization_layer = MiniBatchKMeans(
        n_clusters=codebook_width,
        n_features=embedding_dim,
        distance_function=SquaredEuclideanDistance(),
        initializer=KMeansPlusPlusInitInitializer(
            n_clusters=codebook_width,
            distance_function=SquaredEuclideanDistance(),
            initialize_on_cpu=False
        ),
        init_buffer_size=init_buffer_size,
        optimizer=None,
    )
    
    model = ResidualQuantization(
        n_layers=num_hierarchies,
        quantization_layer=quantization_layer,
        quantization_layer_list=None,
        init_buffer_size=init_buffer_size,
        training_loop_function=None,
        quantization_loss_weight=1.0,
        reconstruction_loss_weight=0.0,
        normalize_residuals=True,
        optimizer=None,
        scheduler=None,
        train_layer_wise=True,
        track_residuals=True,
        verbose=False,  # Less verbose for inference
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    
    # Create test embeddings
    print("Creating test embeddings...")
    num_test_items = 50
    test_embeddings = torch.randn(num_test_items, embedding_dim).to(device)
    
    # Add some structure
    cluster_centers = torch.randn(3, embedding_dim) * 2
    for i in range(3):
        start_idx = i * (num_test_items // 3)
        end_idx = (i + 1) * (num_test_items // 3) if i < 2 else num_test_items
        test_embeddings[start_idx:end_idx] = cluster_centers[i] + torch.randn(end_idx - start_idx, embedding_dim) * 0.3
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        model_output = model.predict_step(test_embeddings)
        predictions = model_output.predictions
        item_ids = model_output.keys
    
    # Display results
    print(f"\nInference completed!")
    print(f"Generated {len(predictions)} semantic IDs")
    print(f"Shape: {predictions.shape}")
    print(f"Hierarchies: {predictions.shape[1]}")
    
    # Show statistics
    print(f"\nStatistics:")
    for hierarchy in range(predictions.shape[1]):
        unique_values = torch.unique(predictions[:, hierarchy])
        print(f"  Hierarchy {hierarchy}: {len(unique_values)} unique values")
    
    # Show sample predictions
    print(f"\nSample predictions:")
    for i in range(min(10, len(predictions))):
        print(f"  Item {item_ids[i]}: {predictions[i].tolist()}")
    
    # Check for similar items
    print(f"\nChecking for similar items...")
    unique_combinations = torch.unique(predictions, dim=0)
    print(f"Unique semantic ID combinations: {len(unique_combinations)}/{len(predictions)}")
    
    # Find items with same semantic ID
    for i, combo in enumerate(unique_combinations[:5]):  # Show first 5 combinations
        mask = (predictions == combo).all(dim=1)
        similar_items = torch.where(mask)[0].tolist()
        if len(similar_items) > 1:
            print(f"  Combination {combo.tolist()}: Items {similar_items}")
    
    print("\nInference example completed successfully!")


if __name__ == "__main__":
    main()
