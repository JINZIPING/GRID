import torch
import torch.nn as nn
import torch.nn.functional as F
from .rkmeans_vq import RKMeansVectorQuantizer


class RKMeansResidualQuantizer(nn.Module):
    """
    Residual Quantizer for RKMeans.
    Implements hierarchical quantization using multiple K-means layers.
    Pure clustering approach without VAE components.
    """

    def __init__(self, n_e_list, e_dim, beta=0.25, kmeans_init=True, kmeans_iters=10,
                 init_buffer_size=1000, initialize_on_cpu=False, normalize_residuals=True):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.init_buffer_size = init_buffer_size
        self.initialize_on_cpu = initialize_on_cpu
        self.normalize_residuals = normalize_residuals
        self.num_quantizers = len(n_e_list)

        # Create quantization layers
        self.vq_layers = nn.ModuleList([
            RKMeansVectorQuantizer(
                n_e=n_e,
                e_dim=e_dim,
                beta=self.beta,
                kmeans_init=self.kmeans_init,
                kmeans_iters=self.kmeans_iters,
                init_buffer_size=self.init_buffer_size,
                initialize_on_cpu=self.initialize_on_cpu
            )
            for n_e in n_e_list
        ])

    def get_codebook(self):
        """Get all codebooks from all quantization layers."""
        all_codebooks = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebooks.append(codebook)
        return torch.stack(all_codebooks)

    def forward(self, x, use_sk=True):
        """
        Forward pass through all quantization layers.
        
        Args:
            x: Input tensor of shape (batch_size, e_dim)
            use_sk: Whether to use Sinkhorn algorithm (not used in K-means)
        
        Returns:
            x_q: Quantized vectors
            mean_loss: Average quantization loss across all layers
            all_indices: All cluster assignments from all layers
        """
        all_losses = []
        all_indices = []
        
        x_q = torch.zeros_like(x)
        residual = x
        
        for i, quantizer in enumerate(self.vq_layers):
            # Normalize residuals if specified
            if self.normalize_residuals and i > 0:
                residual = F.normalize(residual, p=2, dim=-1)
            
            # Quantize current residual
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            
            # Update residual
            residual = residual - x_res
            x_q = x_q + x_res
            
            all_losses.append(loss)
            all_indices.append(indices)
        
        # Compute mean loss across all layers
        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        
        return x_q, mean_loss, all_indices

    def get_indices(self, x, use_sk=False):
        """
        Get cluster assignments for input without computing gradients.
        
        Args:
            x: Input tensor of shape (batch_size, e_dim)
            use_sk: Whether to use Sinkhorn algorithm (not used in K-means)
        
        Returns:
            all_indices: All cluster assignments from all layers
        """
        with torch.no_grad():
            _, _, all_indices = self.forward(x, use_sk=use_sk)
        return all_indices

    def reset_cluster_counts(self):
        """Reset cluster counts for all quantization layers."""
        for quantizer in self.vq_layers:
            quantizer.reset_cluster_counts()