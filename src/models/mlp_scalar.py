"""
Lightweight MLP scalar surrogate model.

Maps geometry and flow parameters to scalar outputs:
- Strouhal number (St_peak)
- Peak amplitude (A_peak)
- Quality factor (Q)
- Mean force coefficients (mean_Cd, mean_Cl, mean_Cm)
"""

import torch
import torch.nn as nn
from typing import List, Optional


def get_device() -> torch.device:
    """
    Get the best available device (MPS for Apple Silicon, CUDA for NVIDIA, CPU otherwise).
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class MLPScalarSurrogate(nn.Module):
    """
    Multi-layer perceptron for scalar regression.
    
    Maps input features (D, H, H/D, Re, sin(aoa), cos(aoa)) to
    output targets (St_peak, A_peak, Q, mean_Cd, mean_Cl, mean_Cm).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 64, 32],
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of output targets
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
            dropout: Dropout probability (0 = no dropout)
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'elu':
            act_fn = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            
            layers.append(act_fn())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (same as forward, for sklearn-style interface).
        """
        return self.forward(x)


class FeatureNormalizer:
    """
    Normalize features to zero mean and unit variance.
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X: torch.Tensor) -> 'FeatureNormalizer':
        """
        Compute mean and std from training data.
        
        Args:
            X: Training features (n_samples, n_features)
            
        Returns:
            self
        """
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True)
        # Avoid division by zero
        self.std = torch.where(self.std > 1e-8, self.std, torch.ones_like(self.std))
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Normalize features.
        
        Args:
            X: Features to normalize
            
        Returns:
            Normalized features
        """
        if self.mean is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fit and transform in one step.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize features.
        """
        if self.mean is None:
            raise ValueError("Normalizer not fitted.")
        return X_norm * self.std + self.mean
    
    def state_dict(self):
        """Return state for saving."""
        return {'mean': self.mean, 'std': self.std}
    
    def load_state_dict(self, state):
        """Load state from checkpoint."""
        self.mean = state['mean']
        self.std = state['std']


def create_mlp_model(
    n_features: int,
    n_targets: int,
    hidden_dims: Optional[List[int]] = None,
    **kwargs
) -> MLPScalarSurrogate:
    """
    Factory function to create MLP model with sensible defaults.
    
    Args:
        n_features: Number of input features
        n_targets: Number of output targets
        hidden_dims: Hidden layer dimensions (default: [64, 64, 32])
        **kwargs: Additional arguments for MLPScalarSurrogate
        
    Returns:
        MLPScalarSurrogate instance
    """
    if hidden_dims is None:
        hidden_dims = [64, 64, 32]
    
    return MLPScalarSurrogate(
        input_dim=n_features,
        output_dim=n_targets,
        hidden_dims=hidden_dims,
        **kwargs
    )
