import torch
import torch.nn as nn
import torch.nn.functional as F


class RKMeansVectorQuantizer(nn.Module):
    """
    Single layer K-means vector quantizer for RKMeans.
    Pure clustering approach without VAE components.
    """

    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=True, kmeans_iters=10,
                 init_buffer_size=1000, initialize_on_cpu=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.init_buffer_size = init_buffer_size
        self.initialize_on_cpu = initialize_on_cpu

        # Initialize centroids
        self.centroids = nn.Parameter(torch.zeros(self.n_e, self.e_dim), requires_grad=True)
        
        # Initialize buffer for K-means++ initialization
        self.init_buffer = torch.tensor([])
        self.is_initialized = False
        self.cluster_counts = torch.zeros(self.n_e)

        if not kmeans_init:
            self.is_initialized = True
            self.centroids.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.centroids.data.zero_()

    def get_codebook(self):
        """Get the current codebook (centroids)."""
        return self.centroids

    def get_codebook_entry(self, indices, shape=None):
        """Get quantized vectors from indices."""
        z_q = self.centroids[indices]
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def _buffer_points(self, batch):
        """Buffer points for K-means++ initialization."""
        batch = batch.detach()
        n_to_add = min(self.init_buffer_size - self.init_buffer.shape[0], batch.shape[0])
        
        if self.init_buffer.numel() == 0:
            self.init_buffer = batch[:n_to_add].clone()
        else:
            self.init_buffer = torch.cat([self.init_buffer, batch[:n_to_add]], dim=0)

    def _init_centroids(self, buffer):
        """Initialize centroids using K-means++ algorithm."""
        if buffer.shape[0] < self.n_e:
            raise ValueError(f"Buffer size {buffer.shape[0]} is less than the number of clusters {self.n_e}.")
        
        centroids = self._kmeans_plus_plus_init(buffer, self.n_e, self.initialize_on_cpu)
        self.centroids.data.copy_(centroids)
        self.is_initialized = True

    def _kmeans_plus_plus_init(self, data, n_clusters, initialize_on_cpu=False):
        """K-means++ initialization algorithm."""
        if initialize_on_cpu:
            data = data.cpu()
        
        n_samples, n_features = data.shape
        device = data.device
        dtype = data.dtype
        
        # Initialize centroids
        centroids = torch.zeros(n_clusters, n_features, device=device, dtype=dtype)
        
        # Choose first centroid randomly
        first_idx = torch.randint(0, n_samples, (1,), device=device)
        centroids[0] = data[first_idx]
        
        # Choose remaining centroids using K-means++ algorithm
        for i in range(1, n_clusters):
            # Compute distances to existing centroids
            distances = torch.cdist(data, centroids[:i], p=2)
            min_distances = distances.min(dim=1)[0]
            
            # Compute probabilities (squared distances)
            probabilities = min_distances ** 2
            probabilities = probabilities / probabilities.sum()
            
            # Choose next centroid based on probabilities
            next_idx = torch.multinomial(probabilities, 1)
            centroids[i] = data[next_idx]
        
        return centroids

    def _compute_distances(self, x, centroids):
        """Compute squared Euclidean distances between x and centroids."""
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        centroids_norm = (centroids ** 2).sum(dim=1, keepdim=True)
        xy = torch.mm(x, centroids.t())
        distances = x_norm + centroids_norm.t() - 2 * xy
        return distances

    def forward(self, x, use_sk=True):
        """
        Forward pass of the K-means quantizer.
        
        Args:
            x: Input tensor of shape (batch_size, e_dim)
            use_sk: Whether to use Sinkhorn algorithm (not used in K-means)
        
        Returns:
            x_q: Quantized vectors
            loss: Quantization loss
            indices: Cluster assignments
        """
        # Flatten input
        latent = x.view(-1, self.e_dim)
        
        # Initialize centroids if needed
        if not self.is_initialized and self.training:
            self._buffer_points(latent)
            if self.init_buffer.shape[0] >= self.init_buffer_size:
                self._init_centroids(self.init_buffer)
                self.init_buffer = torch.tensor([])

        # Compute distances to centroids
        distances = self._compute_distances(latent, self.centroids)
        
        # Assign to closest centroids
        indices = torch.argmin(distances, dim=-1)
        
        # Get quantized vectors
        x_q = self.centroids[indices].view(x.shape)
        
        # Compute quantization loss
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss
        
        # Preserve gradients
        x_q = x + (x_q - x).detach()
        
        # Reshape indices to match input shape
        indices = indices.view(x.shape[:-1])
        
        return x_q, loss, indices

    def update_centroids(self, assignments, embeddings):
        """Update centroids based on assignments and embeddings."""
        if not self.training:
            return
            
        # Convert assignments to one-hot
        assignments_one_hot = F.one_hot(assignments, self.n_e).float()
        
        # Update cluster counts
        batch_cluster_counts = assignments_one_hot.sum(dim=0)
        self.cluster_counts += batch_cluster_counts.to(self.cluster_counts.device)
        
        # Update centroids
        batch_cluster_sums = torch.mm(assignments_one_hot.t(), embeddings)
        
        # Only update centroids that have assignments
        mask = batch_cluster_counts > 0
        if mask.any():
            self.centroids.data[mask] = batch_cluster_sums[mask] / batch_cluster_counts[mask].unsqueeze(1)