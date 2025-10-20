#!/usr/bin/env python3
"""
PyTorch-only training script for semantic ID generation using Kmeans clustering.
This script removes Lightning dependency and implements training using pure PyTorch.
"""

import os
import time
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import hydra
import rootutils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.custom_hydra_resolvers import *
from src.components.distance_functions import SquaredEuclideanDistance
from src.components.clustering_initializers import KMeansPlusPlusInitInitializer
from src.components.loss_functions import WeightedSquaredError
from src.models.modules.clustering.mini_batch_kmeans import MiniBatchKMeans
from src.modules.clustering.residual_quantization import ResidualQuantization

torch.set_float32_matmul_precision("medium")


class PyTorchOnlyTrainer:
    """PyTorch-only trainer that replaces Lightning functionality."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create output directory
        self.output_dir = Path(cfg.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def setup_model(self):
        """Initialize the ResidualQuantization model for Kmeans clustering."""
        print("Setting up ResidualQuantization model...")
        
        # Load embeddings
        embeddings = None
        if self.cfg.embedding_path:
            embeddings = torch.load(self.cfg.embedding_path, map_location=self.device)
            print(f"Loaded embeddings with shape: {embeddings.shape}")
        
        # Create quantization layer (MiniBatchKMeans)
        quantization_layer = MiniBatchKMeans(
            n_clusters=self.cfg.codebook_width,
            n_features=self.cfg.embedding_dim,
            distance_function=SquaredEuclideanDistance(),
            initializer=KMeansPlusPlusInitInitializer(
                n_clusters=self.cfg.codebook_width,
                distance_function=SquaredEuclideanDistance(),
                initialize_on_cpu=False
            ),
            init_buffer_size=self.cfg.model.init_buffer_size,
            optimizer=None,  # Will be handled by ResidualQuantization
        )
        
        # Create ResidualQuantization model
        self.model = ResidualQuantization(
            n_layers=self.cfg.num_hierarchies,
            quantization_layer=quantization_layer,
            quantization_layer_list=None,
            init_buffer_size=self.cfg.model.init_buffer_size,
            training_loop_function=None,  # Will implement custom training loop
            quantization_loss_weight=1.0,
            reconstruction_loss_weight=0.0,
            normalize_residuals=self.cfg.model.normalize_residuals,
            optimizer=optim.SGD,
            scheduler=None,
            train_layer_wise=self.cfg.model.train_layer_wise,
            track_residuals=self.cfg.model.track_residuals,
            verbose=self.cfg.model.verbose,
        )
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.5
        )
        
        print(f"ResidualQuantization model setup complete. Device: {self.device}")
        
    def setup_data(self):
        """Setup data loaders for ResidualQuantization training."""
        print("Setting up data loaders...")
        
        # Load embeddings if available
        if self.cfg.embedding_path:
            embeddings = torch.load(self.cfg.embedding_path, map_location=self.device)
            print(f"Using loaded embeddings with shape: {embeddings.shape}")
        else:
            # Create dummy embeddings for demonstration
            print("Creating dummy embeddings...")
            embeddings = torch.randn(10000, self.cfg.embedding_dim)
        
        # Create dataset
        batch_size = 2048  # From rkmeans_train_flat.yaml
        dummy_dataset = torch.utils.data.TensorDataset(embeddings)
        
        self.train_loader = DataLoader(
            dummy_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Create validation loader
        val_embeddings = embeddings[:1000] if embeddings.size(0) > 1000 else embeddings
        val_dataset = torch.utils.data.TensorDataset(val_embeddings)
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=256,  # From rkmeans_train_flat.yaml
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            persistent_workers=True
        )
        
        print("Data loaders setup complete")
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (embeddings,) in enumerate(self.train_loader):
            embeddings = embeddings.to(self.device)
            
            # Create ItemData-like object for the model
            class SimpleItemData:
                def __init__(self, embeddings):
                    self.transformed_features = {"input_embedding": embeddings}
                    self.item_ids = list(range(embeddings.size(0)))
            
            item_data = SimpleItemData(embeddings)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get model output using model_step
            cluster_ids, all_residuals, quantization_loss, reconstruction_loss = self.model.model_step(item_data)
            
            # Compute total loss
            loss = quantization_loss + reconstruction_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if self.global_step % 10 == 0:  # From rkmeans_train_flat.yaml log_every_n_steps
                print(
                    f"Epoch {self.epoch}, Step {self.global_step}, "
                    f"Loss: {loss.item():.4f}, Avg Loss: {total_loss/num_batches:.4f}"
                )
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for embeddings, in self.val_loader:
                embeddings = embeddings.to(self.device)
                
                # Forward pass
                cluster_ids, all_residuals, quantized_embeddings, quantization_loss = self.model.forward(embeddings)
                
                total_loss += quantization_loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch:03d}_step_{self.global_step:06d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved at {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        command_line_logger.info("Starting training...")
        
        # Load checkpoint if specified
        if self.cfg.get("ckpt_path"):
            self.load_checkpoint(self.cfg.ckpt_path)
        
        # Training loop
        for epoch in range(self.epoch, self.cfg.trainer.max_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Log metrics
            command_line_logger.info(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.cfg.callbacks.early_stopping.patience:
                command_line_logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        command_line_logger.info("Training completed!")


def train_pytorch_only(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """PyTorch-only training function."""
    
    # Set random seed
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
    
    # Initialize trainer
    trainer = PyTorchOnlyTrainer(cfg)
    
    # Setup model and data
    trainer.setup_model()
    trainer.setup_data()
    
    # Start training
    trainer.train()
    
    # Return metrics (simplified)
    metrics = {
        "best_val_loss": trainer.best_val_loss,
        "final_epoch": trainer.epoch,
        "total_steps": trainer.global_step
    }
    
    return metrics, {}


@hydra.main(version_base="1.3", config_path="../configs/experiment", config_name="rkmeans_train_flat")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for PyTorch-only Kmeans training."""
    # Apply extra utilities
    extras(cfg)
    
    # Run training
    metrics, _ = train_pytorch_only(cfg)
    
    print(f"Training completed with metrics: {metrics}")
    return metrics.get("best_val_loss")


if __name__ == "__main__":
    main()