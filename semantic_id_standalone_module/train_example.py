#!/usr/bin/env python3
"""
Simple training example for semantic ID module.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from semantic_id_module import (
    ResidualQuantization, 
    MiniBatchKMeans, 
    SquaredEuclideanDistance, 
    KMeansPlusPlusInitInitializer
)


def main():
    """Simple training example."""
    print("Semantic ID Training Example")
    print("=" * 40)
    
    # Parameters (matching original config)
    num_items = 2000
    embedding_dim = 2048
    num_hierarchies = 3
    codebook_width = 256
    batch_size = 64
    num_epochs = 2
    init_buffer_size = 3072  # Match original config
    
    # Create dummy data
    print(f"Creating {num_items} dummy embeddings...")
    embeddings = torch.randn(num_items, embedding_dim)
    
    # Add some structure for better clustering
    cluster_centers = torch.randn(5, embedding_dim) * 2
    cluster_assignments = torch.randint(0, 5, (num_items,))
    for i in range(5):
        mask = cluster_assignments == i
        embeddings[mask] = cluster_centers[i] + torch.randn(mask.sum(), embedding_dim) * 0.5
    
    # Create dataset
    dataset = TensorDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
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
        optimizer=optim.SGD,
        scheduler=None,
        train_layer_wise=True,
        track_residuals=True,
        verbose=True,
    )
    
    # Setup device and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (batch_embeddings,) in enumerate(dataloader):
            batch_embeddings = batch_embeddings.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            cluster_ids, all_residuals, quantization_loss, reconstruction_loss = model.model_step(batch_embeddings)
            
            # Compute loss
            loss = quantization_loss + reconstruction_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    print("Training completed!")
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    test_embeddings = torch.randn(100, embedding_dim).to(device)
    
    with torch.no_grad():
        model_output = model.predict_step(test_embeddings)
        predictions = model_output.predictions
        item_ids = model_output.keys
        
        print(f"Generated {len(predictions)} semantic IDs")
        print(f"Shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[:5].tolist()}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
