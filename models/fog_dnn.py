"""
Fog/Server-Level Deep Neural Network for Multi-Class Intrusion Detection.
Phase 5: PyTorch Fog Classifier for 15-Class Attack Categorization.

Architecture:
    Input (51 features) -> Configurable Dense Layers -> ReLU -> Dropout -> Output (15 classes)
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FogDNN(nn.Module):
    """
    Configurable PyTorch Feed-Forward Deep Neural Network for Fog-level
    multi-class intrusion classification on IIoT network flows.
    """

    def __init__(
        self,
        input_dim: int = 51,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 15,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = False,
        activation: str = "relu",
    ):
        """
        Initialize the FogDNN model.

        Args:
            input_dim: Number of input features (default: 51).
            hidden_dims: List of integers specifying hidden layer widths.
            num_classes: Number of output attack classes (default: 15).
            dropout_rate: Dropout probability for regularization.
            use_batch_norm: Whether to apply BatchNorm1d after each hidden layer.
            activation: Activation function ('relu' or 'leaky_relu').
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation.lower()

        # Build layers
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for i, h_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))

            if self.activation_name == "leaky_relu":
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            else:
                layers.append(nn.ReLU())

            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))

            prev_dim = h_dim

        # Final classification head (raw logits)
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized class logits.

        Args:
            x: Input tensor of shape (batch_size, 51).

        Returns:
            Logits tensor of shape (batch_size, 15).
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute softmax probabilities for input tensor.

        Args:
            x: Input tensor of shape (batch_size, 51).

        Returns:
            Probability tensor of shape (batch_size, 15).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute predicted class indices for input tensor.

        Args:
            x: Input tensor of shape (batch_size, 51).

        Returns:
            Class index tensor of shape (batch_size,).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)

    def get_model_size_kb(self) -> float:
        """Calculate total parameter and buffer size in kilobytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024.0

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, filepath: Union[str, os.PathLike]) -> None:
        """
        Save model weights and architectural configuration.

        Args:
            filepath: Destination path for the .pth checkpoint.
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "init_args": {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "num_classes": self.num_classes,
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
                "activation": self.activation_name,
            },
            "num_parameters": self.count_parameters(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(checkpoint, filepath)

    @classmethod
    def load(
        cls,
        filepath: Union[str, os.PathLike],
        device: Optional[torch.device] = None,
    ) -> "FogDNN":
        """
        Load model from checkpoint file.

        Args:
            filepath: Path to the .pth checkpoint.
            device: Target device (CPU, CUDA, MPS).

        Returns:
            Re-instantiated FogDNN model with loaded weights.
        """
        if device is None:
            device = torch.device("cpu")

        checkpoint = torch.load(filepath, map_location=device)
        init_args = checkpoint["init_args"]
        model = cls(**init_args)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
