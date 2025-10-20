#!/usr/bin/env python3
"""
Test script for semantic ID training and inference with dummy data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import pickle
import json


def create_dummy_embeddings(num_items=10000, embedding_dim=2048, save_path="dummy_embeddings.pt"):
    """Create dummy embeddings and save them."""
    print(f"Creating dummy embeddings: {num_items} items, {embedding_dim} dimensions")
    
    # Generate random embeddings
    embeddings = torch.randn(num_items, embedding_dim)
    
    # Add some structure to make clustering more meaningful
    # Create 5 clusters with different centers
    cluster_centers = torch.randn(5, embedding_dim) * 2
    cluster_assignments = torch.randint(0, 5, (num_items,))
    
    for i in range(5):
        mask = cluster_assignments == i
        embeddings[mask] = cluster_centers[i] + torch.randn(mask.sum(), embedding_dim) * 0.5
    
    # Save embeddings
    torch.save(embeddings, save_path)
    print(f"Saved embeddings to {save_path}")
    
    return embeddings


def create_item_embedding_mapping(embeddings, save_path="item_embedding_mapping.json"):
    """Create item ID to embedding mapping."""
    print("Creating item-embedding mapping...")
    
    mapping = {}
    for i, embedding in enumerate(embeddings):
        mapping[str(i)] = {
            "item_id": i,
            "embedding_idx": i,
            "category": f"category_{i % 10}",  # 10 categories
            "price": float(np.random.uniform(1.0, 100.0)),
            "rating": float(np.random.uniform(1.0, 5.0))
        }
    
    # Save mapping
    with open(save_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Saved item-embedding mapping to {save_path}")
    return mapping


def test_training_with_dummy_data():
    """Test training with dummy data."""
    print("=" * 50)
    print("TESTING TRAINING WITH DUMMY DATA")
    print("=" * 50)
    
    # Parameters (matching original config)
    num_items = 5000
    embedding_dim = 2048
    num_hierarchies = 3
    codebook_width = 256
    batch_size = 64
    num_epochs = 3
    init_buffer_size = 3072  # Match original config
    
    # Create dummy data
    embeddings = create_dummy_embeddings(num_items, embedding_dim)
    mapping = create_item_embedding_mapping(embeddings)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = TensorDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Import the standalone module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.semantic_id_module import (
        SquaredEuclideanDistance, 
        KMeansPlusPlusInitInitializer, 
        WeightedSquaredError,
        MiniBatchKMeans,
        ResidualQuantization
    )
    
    # Create quantization layer
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
    
    # Create ResidualQuantization model
    model = ResidualQuantization(
        n_layers=num_hierarchies,
        quantization_layer=quantization_layer,
        quantization_layer_list=None,
        init_buffer_size=init_buffer_size,
        training_loop_function=None,
        quantization_loss_weight=1.0,
        reconstruction_loss_weight=0.0,
        normalize_residuals=True,
        optimizer=optim.SGD,
        scheduler=None,
        train_layer_wise=True,
        track_residuals=True,
        verbose=True,
    )
    
    model = model.to(device)
    
    # Setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    
    # Training loop
    print("\nStarting training...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (batch_embeddings,) in enumerate(dataloader):
            batch_embeddings = batch_embeddings.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            cluster_ids, all_residuals, quantization_loss, reconstruction_loss = model.model_step(batch_embeddings)
            
            # Compute total loss
            loss = quantization_loss + reconstruction_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    print("Training completed!")
    
    # Save model
    model_path = "trained_semantic_id_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_hierarchies': num_hierarchies,
        'codebook_width': codebook_width,
        'embedding_dim': embedding_dim,
    }, model_path)
    print(f"Model saved to {model_path}")
    
    return model, embeddings, mapping


def test_inference_with_dummy_data(model, embeddings, mapping):
    """Test inference with dummy data."""
    print("\n" + "=" * 50)
    print("TESTING INFERENCE WITH DUMMY DATA")
    print("=" * 50)
    
    device = next(model.parameters()).device
    
    # Create test data (subset of training data)
    test_size = 1000
    test_indices = torch.randperm(len(embeddings))[:test_size]
    test_embeddings = embeddings[test_indices]
    test_item_ids = test_indices.tolist()
    
    print(f"Testing with {test_size} items")
    
    # Run inference
    model.eval()
    all_predictions = []
    all_item_ids = []
    
    batch_size = 64
    with torch.no_grad():
        for i in range(0, test_size, batch_size):
            batch_embeddings = test_embeddings[i:i+batch_size].to(device)
            batch_item_ids = test_item_ids[i:i+batch_size]
            
            # Get predictions
            model_output = model.predict_step(batch_embeddings)
            predictions = model_output.predictions
            item_ids = model_output.keys
            
            all_predictions.append(predictions.cpu())
            all_item_ids.extend(item_ids)
            
            if (i // batch_size) % 5 == 0:
                print(f"Processed {i+len(batch_embeddings)}/{test_size} items")
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    
    print(f"\nInference completed!")
    print(f"Generated {all_predictions.shape[0]} predictions")
    print(f"Prediction shape: {all_predictions.shape}")
    print(f"Number of hierarchies: {all_predictions.shape[1]}")
    
    # Show some statistics
    print(f"\nPrediction statistics:")
    for hierarchy in range(all_predictions.shape[1]):
        unique_values = torch.unique(all_predictions[:, hierarchy])
        print(f"  Hierarchy {hierarchy}: {len(unique_values)} unique values, range [{all_predictions[:, hierarchy].min()}, {all_predictions[:, hierarchy].max()}]")
    
    # Show sample predictions
    print(f"\nSample predictions (first 10 items):")
    for i in range(min(10, len(all_predictions))):
        item_id = all_item_ids[i]
        prediction = all_predictions[i].tolist()
        print(f"  Item {item_id}: {prediction}")
    
    # Save predictions
    predictions_data = {
        'predictions': all_predictions,
        'item_ids': all_item_ids,
        'metadata': {
            'num_hierarchies': all_predictions.shape[1],
            'codebook_width': 256,
            'embedding_dim': 2048,
            'total_samples': all_predictions.shape[0]
        }
    }
    
    # Save as pickle
    with open('semantic_id_predictions.pkl', 'wb') as f:
        pickle.dump(predictions_data, f)
    
    # Save as PyTorch tensor
    torch.save(all_predictions, 'semantic_id_predictions.pt')
    
    print(f"\nPredictions saved to:")
    print(f"  - semantic_id_predictions.pkl")
    print(f"  - semantic_id_predictions.pt")
    
    return all_predictions, all_item_ids


def test_semantic_id_quality(predictions, item_ids, mapping):
    """Test the quality of generated semantic IDs."""
    print("\n" + "=" * 50)
    print("TESTING SEMANTIC ID QUALITY")
    print("=" * 50)
    
    # Check uniqueness
    unique_combinations = torch.unique(predictions, dim=0)
    print(f"Unique semantic ID combinations: {len(unique_combinations)}/{len(predictions)}")
    print(f"Uniqueness ratio: {len(unique_combinations)/len(predictions):.4f}")
    
    # Check distribution across hierarchies
    print(f"\nDistribution across hierarchies:")
    for hierarchy in range(predictions.shape[1]):
        values, counts = torch.unique(predictions[:, hierarchy], return_counts=True)
        print(f"  Hierarchy {hierarchy}: {len(values)} unique values")
        print(f"    Most common: {values[counts.argmax()]} (count: {counts.max()})")
        print(f"    Least common: {values[counts.argmin()]} (count: {counts.min()})")
    
    # Check if similar items get similar semantic IDs
    print(f"\nChecking semantic ID consistency...")
    
    # Sample some items and check their neighbors
    sample_size = 100
    sample_indices = torch.randperm(len(predictions))[:sample_size]
    
    similarities = []
    for i in sample_indices:
        item_pred = predictions[i]
        # Find items with similar semantic IDs (Hamming distance)
        hamming_distances = (predictions != item_pred.unsqueeze(0)).sum(dim=1)
        similar_items = torch.where(hamming_distances <= 1)[0]  # At most 1 different hierarchy
        
        if len(similar_items) > 1:
            similarities.append(len(similar_items))
    
    if similarities:
        avg_similar = np.mean(similarities)
        print(f"Average number of similar items (Hamming distance <= 1): {avg_similar:.2f}")
    
    print("Quality analysis completed!")


def main():
    """Main test function."""
    print("Starting comprehensive semantic ID testing with dummy data...")
    
    try:
        # Test training
        model, embeddings, mapping = test_training_with_dummy_data()
        
        # Test inference
        predictions, item_ids = test_inference_with_dummy_data(model, embeddings, mapping)
        
        # Test quality
        test_semantic_id_quality(predictions, item_ids, mapping)
        
        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        # Summary
        print(f"\nSummary:")
        print(f"- Trained model with {len(embeddings)} items")
        print(f"- Generated semantic IDs for {len(predictions)} test items")
        print(f"- Semantic ID shape: {predictions.shape}")
        print(f"- Model parameters: {sum(p.numel() for p in model.parameters())}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
