"""
Shared test model definitions for ModelBatch tests.
"""

import torch
from torch import nn
from torch.utils.data import TensorDataset


def create_dummy_data(
    num_samples: int = 1000, input_size: int = 784, num_classes: int = 10
):
    """Create dummy classification data for testing."""
    x = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(x, y)


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""

    def __init__(self, input_size=10, hidden_size=5, output_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


class CustomModel(nn.Module):
    """Custom model with nontrivial logic for testing."""

    def __init__(self, input_size=5, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Conditional skip connection
        # [Note: Python conditionals ("if") cannot be parsed by functorch, so `vmap` will error. `torch.where` is a workaround.]
        h = torch.tanh(self.fc1(x))
        h = h + torch.where(x.sum() > 0, x, torch.zeros_like(x))
        return self.fc2(h)


class DeepMLP(nn.Module):
    """Deeper MLP for testing complex models."""

    def __init__(self, input_size=4, output_size=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_size),
        )

    def forward(self, x):
        return self.network(x)


class SimpleCNN(nn.Module):
    """Simple CNN for testing convolutional models."""

    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)


class ImageMLP(nn.Module):
    """Simple MLP for image classification tasks (e.g., CIFAR-10, MNIST)."""

    def __init__(
        self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.layers(x)
