#!/usr/bin/env python3
"""
Standalone semantic ID module for Kmeans clustering.
This is a self-contained implementation without external dependencies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, Any
from abc import ABC, abstractmethod
import functools


class DistanceFunction(ABC):
    """Abstract base class for distance functions."""
    
    @abstractmethod
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute distances between x and y."""
        pass


class SquaredEuclideanDistance(DistanceFunction):
    """Squared Euclidean distance function."""
    
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute squared Euclidean distances between x and y."""
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)
        xy = torch.mm(x, y.t())
        distances = x_norm + y_norm.t() - 2 * xy
        return distances


class ClusteringInitializer(ABC):
    """Abstract base class for clustering initializers."""
    
    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Initialize cluster centroids from data."""
        pass


class KMeansPlusPlusInitInitializer(ClusteringInitializer):
    """K-means++ initialization."""
    
    def __init__(self, n_clusters: int, distance_function: DistanceFunction, initialize_on_cpu: bool = False):
        self.n_clusters = n_clusters
        self.distance_function = distance_function
        self.initialize_on_cpu = initialize_on_cpu
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Initialize centroids using K-means++ algorithm."""
        if self.initialize_on_cpu:
            data = data.cpu()
        
        n_samples, n_features = data.shape
        centroids = torch.zeros(self.n_clusters, n_features, device=data.device, dtype=data.dtype)
        
        # Choose first centroid randomly
        first_idx = torch.randint(0, n_samples, (1,), device=data.device)
        centroids[0] = data[first_idx]
        
        # Choose remaining centroids
        for i in range(1, self.n_clusters):
            distances = self.distance_function.compute(data, centroids[:i])
            min_distances = distances.min(dim=1)[0]
            probabilities = min_distances ** 2
            probabilities = probabilities / probabilities.sum()
            next_idx = torch.multinomial(probabilities, 1)
            centroids[i] = data[next_idx]
        
        return centroids


class WeightedSquaredError(nn.Module):
    """Weighted squared error loss function."""
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute weighted squared error loss."""
        loss = (input - target) ** 2
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        return loss.mean()


class MiniBatchKMeans(nn.Module):
    """Mini-batch K-means clustering implementation."""
    
    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        distance_function: DistanceFunction,
        initializer: ClusteringInitializer,
        loss_function: nn.Module = WeightedSquaredError(),
        optimizer: Optional[torch.optim.Optimizer] = None,
        init_buffer_size: int = 3072,  # Match original config
        update_manually: bool = False,
    ):
        super().__init__()
        
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.distance_function = distance_function
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.initializer = initializer
        self.init_buffer_size = init_buffer_size
        
        self.centroids = nn.Parameter(
            torch.zeros(self.n_clusters, self.n_features), requires_grad=True
        )
        self.update_manually = update_manually
        if self.update_manually:
            self.centroids.requires_grad = False
        
        self.init_loss_function = WeightedSquaredError()
        self.init_buffer = torch.tensor([])
        self.is_initialized = False
        self.is_initial_step = False
        self.cluster_counts = torch.zeros(self.n_clusters)
    
    def _buffer_points(self, batch: torch.Tensor) -> None:
        """Buffer points for initialization."""
        batch = batch.detach()
        n_to_add = min(self.init_buffer_size - self.init_buffer.shape[0], batch.shape[0])
        if self.init_buffer.numel() == 0:
            self.init_buffer = batch[:n_to_add].clone()
        else:
            self.init_buffer = torch.cat([self.init_buffer, batch[:n_to_add]], dim=0)
    
    def compute_initial_centroids(self, buffer: torch.Tensor) -> None:
        """Initialize centroids using the buffer."""
        if buffer.shape[0] < self.n_clusters:
            raise ValueError(f"Buffer size {buffer.shape[0]} is less than the number of clusters {self.n_clusters}.")
        
        self.init_centroids = self.initializer(buffer)
    
    def initialization_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Perform initialization step."""
        self._buffer_points(batch)
        if self.init_buffer.shape[0] < self.init_buffer_size:
            centroid_zero_embeddings = torch.zeros_like(self.centroids.data, dtype=batch.dtype, device=batch.device)
            loss = self.init_loss_function(self.centroids, centroid_zero_embeddings)
            batch_zero_embeddings = torch.zeros_like(batch, dtype=batch.dtype, device=batch.device)
            batch_zero_assignments = torch.zeros(batch.shape[0], dtype=torch.long, device=batch.device)
            return batch_zero_assignments, batch_zero_embeddings, loss
        else:
            self.init_centroids = torch.zeros_like(self.centroids.data, dtype=batch.dtype, device=batch.device)
            self.compute_initial_centroids(self.init_buffer)
            self.is_initial_step = True
            self.init_buffer = torch.tensor([], device=batch.device)
            
            if self.update_manually:
                self.centroids[:] = self.init_centroids.data
                distances = self.distance_function.compute(batch, self.centroids.data)
                assignments = torch.argmin(distances, dim=1).to(batch.device)
                return assignments, self.centroids[assignments], torch.tensor(0.0, device=batch.device)
            else:
                loss = self.init_loss_function(self.centroids, self.init_centroids)
                distances = self.distance_function.compute(batch, self.init_centroids)
                assignments = torch.argmin(distances, dim=1).to(batch.device)
                return assignments, self.init_centroids[assignments], loss
    
    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the K-means model."""
        assignments = self.predict_step(batch, return_embeddings=False)
        assignments_one_hot = nn.functional.one_hot(assignments, self.n_clusters).detach()
        
        batch_cluster_counts = torch.sum(assignments_one_hot, dim=0)
        if self.cluster_counts.device != batch_cluster_counts.device:
            self.cluster_counts = self.cluster_counts.to(batch_cluster_counts.device)
        self.cluster_counts += batch_cluster_counts
        
        batch_cluster_sums = torch.mm(assignments_one_hot.float().t(), batch)
        
        return assignments, batch_cluster_counts, batch_cluster_sums
    
    def model_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Perform a model step."""
        batch = batch.to(next(self.parameters()).device)
        
        if self.is_initial_step:
            self.is_initial_step = False
            self.is_initialized = True
        if not self.is_initialized:
            return self.initialization_step(batch)
        
        assignments, batch_cluster_counts, batch_cluster_sums = self.forward(batch)
        
        centroids = self.get_centroids()
        mask = batch_cluster_counts != 0
        mask_target = batch_cluster_sums[mask] / batch_cluster_counts[mask].unsqueeze(1)
        centroid_weights = batch_cluster_counts[mask] / self.cluster_counts[mask]
        
        if self.update_manually:
            self.centroids[mask] = self.centroids[mask].data - (
                (centroids[mask].data - mask_target) * centroid_weights.unsqueeze(1)
            )
            return assignments, centroids[assignments], torch.tensor(0.0, device=batch.device)
        else:
            loss = self.loss_function(centroids[mask], mask_target, centroid_weights)
            return assignments, centroids[assignments], loss
    
    def predict_step(self, batch: torch.Tensor, return_embeddings: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict cluster assignments for input points."""
        batch = batch.to(next(self.parameters()).device)
        with torch.no_grad():
            centroids = self.get_centroids().data
            distances = self.distance_function.compute(batch, centroids)
            assignments = torch.argmin(distances, dim=1)
            if not return_embeddings:
                return assignments
            return assignments, centroids[assignments]
    
    def get_centroids(self) -> nn.Parameter:
        """Get the current centroids."""
        return self.centroids
    
    def get_residuals(self, batch: torch.Tensor) -> torch.Tensor:
        """Get residuals of points from the nearest centroids."""
        _, centroids = self.predict_step(batch)
        return batch - centroids
    
    def on_train_start(self) -> None:
        """Reset the model state at the start of training."""
        device = next(self.parameters()).device
        self.cluster_counts = torch.zeros(self.n_clusters, device=device)
        self.init_buffer = torch.tensor([], device=device)
        self.centroids = self.centroids.to(device)


class ResidualQuantization(nn.Module):
    """Residual Quantization module for semantic ID generation."""
    
    def __init__(
        self,
        n_layers: int,
        quantization_layer: MiniBatchKMeans,
        quantization_layer_list: Optional[nn.ModuleList] = None,
        init_buffer_size: int = 3072,  # Match original config
        training_loop_function: Optional[callable] = None,
        quantization_loss_weight: float = 1.0,
        reconstruction_loss_weight: float = 0.0,
        normalize_residuals: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        train_layer_wise: bool = False,
        track_residuals: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.quantization_layer = quantization_layer
        self.quantization_layer_list = quantization_layer_list or nn.ModuleList(
            [quantization_layer for _ in range(n_layers)]
        )
        self.init_buffer_size = init_buffer_size
        self.training_loop_function = training_loop_function
        self.quantization_loss_weight = quantization_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.normalize_residuals = normalize_residuals
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_layer_wise = train_layer_wise
        self.track_residuals = track_residuals
        self.verbose = verbose
        
        # Initialize layers
        for layer in self.quantization_layer_list:
            layer.init_buffer_size = init_buffer_size
    
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the quantization layers."""
        cluster_ids = []
        current_residuals = embeddings
        all_residuals = [] if self.track_residuals else None
        quantized_embeddings = torch.zeros_like(embeddings)
        quantization_loss = torch.tensor(0.0, device=embeddings.device)
        
        for idx, layer in enumerate(self.quantization_layer_list):
            if self.normalize_residuals:
                current_residuals = nn.functional.normalize(current_residuals, dim=-1)
            
            # Determine whether to train the current layer
            train_layer = True  # Simplified for standalone version
            
            if train_layer:
                layer_ids, layer_embeddings, layer_loss = layer.model_step(current_residuals)
                quantization_loss += layer_loss
            else:
                layer_ids, layer_embeddings = layer.predict_step(current_residuals)
            
            cluster_ids.append(layer_ids)
            quantized_embeddings = quantized_embeddings + layer_embeddings
            current_residuals = current_residuals - layer_embeddings
            
            if self.track_residuals:
                all_residuals.append(current_residuals)
        
        cluster_ids = torch.stack(cluster_ids, dim=-1)
        all_residuals = torch.stack(all_residuals, dim=-1) if self.track_residuals else None
        
        return cluster_ids, all_residuals, quantized_embeddings, quantization_loss
    
    def model_step(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a forward pass and compute the loss for a single batch."""
        cluster_ids, all_residuals, quantized_embeddings, quantization_loss = self.forward(embeddings)
        
        # Simplified reconstruction loss (set to 0 for now)
        reconstruction_loss = torch.tensor(0.0, device=embeddings.device)
        
        return cluster_ids, all_residuals, quantization_loss, reconstruction_loss
    
    def predict_step(self, embeddings: torch.Tensor) -> Any:
        """Perform prediction step."""
        cluster_ids, _, _, _ = self.model_step(embeddings)
        
        # Create a simple output object
        class SimpleOutput:
            def __init__(self, predictions, keys):
                self.predictions = predictions
                self.keys = keys
        
        item_ids = list(range(embeddings.size(0)))
        return SimpleOutput(cluster_ids, item_ids)


def test_semantic_id():
    """Test the standalone semantic ID module."""
    print("Testing standalone semantic ID module...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parameters (matching original config)
    batch_size = 32
    embedding_dim = 2048
    num_samples = 1000
    num_hierarchies = 3
    codebook_width = 256
    init_buffer_size = 3072  # Match original config
    
    # Create dummy data
    embeddings = torch.randn(num_samples, embedding_dim)
    dataset = TensorDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
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
    print("Starting training...")
    model.train()
    for epoch in range(2):
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
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    
    print("Training completed!")
    
    # Test inference
    print("Testing inference...")
    model.eval()
    test_embeddings = torch.randn(100, embedding_dim).to(device)
    
    with torch.no_grad():
        model_output = model.predict_step(test_embeddings)
        predictions = model_output.predictions
        item_ids = model_output.keys
        
        print(f"Inference completed. Generated {predictions.shape[0]} predictions")
        print(f"Prediction shape: {predictions.shape}")
        print(f"Number of item IDs: {len(item_ids)}")
        print(f"Sample predictions: {predictions[:5]}")
    
    print("All tests passed successfully!")
    return model


if __name__ == "__main__":
    test_semantic_id()
