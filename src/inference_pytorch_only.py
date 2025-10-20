#!/usr/bin/env python3
"""
PyTorch-only inference script for Kmeans clustering and semantic ID generation.
This script removes Lightning dependency and implements inference using pure PyTorch.
"""

import os
import time
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import hydra
import rootutils
import torch
import torch.nn as nn
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


class PyTorchOnlyInference:
    """PyTorch-only inference class that replaces Lightning functionality."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.semantic_id_map = None
        
    def setup_model(self):
        """Initialize the ResidualQuantization model for inference."""
        print("Setting up ResidualQuantization model for inference...")
        
        # Load embeddings
        if self.cfg.embedding_path:
            self.embeddings = torch.load(self.cfg.embedding_path, map_location=self.device)
            print(f"Loaded embeddings with shape: {self.embeddings.shape}")
        else:
            # Create dummy embeddings for demonstration
            print("Creating dummy embeddings...")
            self.embeddings = torch.randn(1000, self.cfg.embedding_dim)
        
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
            init_buffer_size=3072,
            optimizer=None,
        )
        
        # Create ResidualQuantization model
        self.model = ResidualQuantization(
            n_layers=self.cfg.num_hierarchies,
            quantization_layer=quantization_layer,
            quantization_layer_list=None,
            init_buffer_size=3072,
            training_loop_function=None,
            quantization_loss_weight=1.0,
            reconstruction_loss_weight=0.0,
            normalize_residuals=True,
            optimizer=None,
            scheduler=None,
            train_layer_wise=True,
            track_residuals=True,
            verbose=True,
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"ResidualQuantization model setup complete. Device: {self.device}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print("Checkpoint loaded successfully")
        
    def setup_data(self):
        """Setup data loader for inference."""
        print("Setting up data loader for inference...")
        
        # Use the loaded embeddings
        batch_size = 256  # From rkmeans_train_flat.yaml
        dummy_dataset = torch.utils.data.TensorDataset(self.embeddings)
        
        self.data_loader = DataLoader(
            dummy_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )
        
        print("Data loader setup complete")
        
    def predict_batch(self, embeddings):
        """Predict semantic IDs for a batch of embeddings."""
        with torch.no_grad():
            # Create ItemData-like object for the model
            class SimpleItemData:
                def __init__(self, embeddings):
                    self.transformed_features = {"input_embedding": embeddings}
                    self.item_ids = list(range(embeddings.size(0)))
            
            item_data = SimpleItemData(embeddings)
            
            # Get predictions using predict_step
            model_output = self.model.predict_step(item_data)
            
            return model_output
    
    def run_inference(self):
        """Run inference on the dataset."""
        print("Starting inference...")
        
        all_predictions = []
        all_item_ids = []
        
        for batch_idx, (embeddings,) in enumerate(self.data_loader):
            # Predict semantic IDs
            model_output = self.predict_batch(embeddings)
            
            # Extract predictions and item IDs
            predictions = model_output.predictions  # cluster_ids
            item_ids = model_output.keys  # item_ids
            
            # Store results
            all_predictions.append(predictions.cpu())
            all_item_ids.extend(item_ids)
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(self.data_loader)}")
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        
        print(f"Inference completed. Generated {all_predictions.shape[0]} predictions")
        
        return all_predictions, all_item_ids
    
    def save_predictions(self, predictions, item_ids, output_path: str):
        """Save predictions to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle file
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump({
                'predictions': predictions,
                'item_ids': item_ids,
                'metadata': {
                    'num_hierarchies': self.cfg.num_hierarchies,
                    'codebook_width': self.cfg.codebook_width,
                    'embedding_dim': self.cfg.embedding_dim,
                    'total_samples': predictions.shape[0]
                }
            }, f)
        
        print(f"Predictions saved to {output_path}")
        
        # Also save as PyTorch tensor for compatibility
        torch_path = output_path.with_suffix('.pt')
        torch.save(predictions, torch_path)
        print(f"Predictions also saved as tensor to {torch_path}")


def inference_pytorch_only(cfg: DictConfig) -> Dict[str, Any]:
    """PyTorch-only inference function."""
    
    # Set random seed
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
    
    # Initialize inference
    inference = PyTorchOnlyInference(cfg)
    
    # Setup model and data
    inference.setup_model()
    inference.setup_data()
    
    # Load checkpoint if specified
    if cfg.get("ckpt_path"):
        inference.load_checkpoint(cfg.ckpt_path)
    
    # Run inference
    predictions, user_ids = inference.run_inference()
    
    # Save predictions
    output_path = Path(cfg.paths.output_dir) / "pickle" / "merged_predictions_tensor.pt"
    inference.save_predictions(predictions, user_ids, output_path)
    
    # Return results
    results = {
        "predictions_shape": predictions.shape,
        "user_ids_shape": user_ids.shape,
        "output_path": str(output_path)
    }
    
    return results


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for PyTorch-only inference."""
    # Apply extra utilities
    extras(cfg)
    
    # Run inference
    results = inference_pytorch_only(cfg)
    
    command_line_logger.info(f"Inference completed with results: {results}")
    return None


if __name__ == "__main__":
    main()
