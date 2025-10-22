import torch
import torch.nn as nn
from .rkmeans_rq import RKMeansResidualQuantizer


class RKMeans(nn.Module):
    """
    RKMeans model for semantic ID generation.
    Pure clustering-based residual quantization without VAE components.
    This model can be used as a drop-in replacement for RQVAE.
    """

    def __init__(self,
                 in_dim=2048,
                 num_emb_list=None,  # [256, 256, 256] for 3 hierarchies
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=True,
                 kmeans_iters=100,
                 init_buffer_size=3072,
                 initialize_on_cpu=False,
                 normalize_residuals=True,
                 **kwargs):
        super(RKMeans, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list if num_emb_list is not None else [256, 256, 256]
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.init_buffer_size = init_buffer_size
        self.initialize_on_cpu = initialize_on_cpu
        self.normalize_residuals = normalize_residuals

        # Pure residual quantizer (no encoder/decoder needed)
        self.rq = RKMeansResidualQuantizer(
            n_e_list=self.num_emb_list,
            e_dim=self.in_dim,  # Direct quantization of input embeddings
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            init_buffer_size=self.init_buffer_size,
            initialize_on_cpu=self.initialize_on_cpu,
            normalize_residuals=self.normalize_residuals
        )

    def forward(self, x, use_sk=True):
        """
        Forward pass of the RKMeans model.
        Direct residual quantization of input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, in_dim)
            use_sk: Whether to use Sinkhorn algorithm (not used in K-means)
        
        Returns:
            out: Quantized embeddings (same as input for RKMeans)
            rq_loss: Quantization loss
            indices: Cluster assignments for all hierarchies
        """
        # Direct residual quantization of input embeddings
        # No encoder/decoder needed - pure clustering approach
        x_q, rq_loss, indices = self.rq(x, use_sk=use_sk)
        
        return x_q, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        """
        Get cluster assignments without computing gradients.
        
        Args:
            xs: Input embeddings of shape (batch_size, in_dim)
            use_sk: Whether to use Sinkhorn algorithm (not used in K-means)
        
        Returns:
            indices: Cluster assignments for all hierarchies
        """
        # Direct quantization without gradients
        _, _, indices = self.rq(xs, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):
        """
        Compute the total loss for training.
        Pure clustering approach - only quantization loss matters.
        
        Args:
            out: Model output (quantized embeddings)
            quant_loss: Quantization loss from residual quantizer
            xs: Original input (not used for pure clustering)
        
        Returns:
            loss_total: Total loss (only quantization loss)
            loss_recon: Reconstruction loss (always 0 for pure clustering)
        """
        # Pure clustering approach - no reconstruction loss
        # Only quantization loss matters for semantic ID generation
        loss_recon = torch.tensor(0.0, device=quant_loss.device)
        loss_total = self.quant_loss_weight * quant_loss

        return loss_total, loss_recon

    def get_codebook(self):
        """Get all codebooks from all quantization layers."""
        return self.rq.get_codebook()

    def update_centroids(self, assignments, embeddings):
        """Update centroids for all quantization layers."""
        self.rq.update_centroids(assignments, embeddings)

    def get_model_info(self):
        """Get model information for debugging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.in_dim,
            'num_hierarchies': len(self.num_emb_list),
            'codebook_widths': self.num_emb_list,
            'init_buffer_size': self.init_buffer_size,
            'normalize_residuals': self.normalize_residuals,
            'model_type': 'Pure Clustering Residual Quantizer'
        }
        
        return info
