"""
Shared test model definitions for ModelBatch tests.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""
    
    def __init__(self, input_size=10, hidden_size=5, output_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class CustomModel(nn.Module):
    """Custom model with nontrivial logic for testing."""
    
    def __init__(self, input_size=5, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, output_size)
    
    def forward(self, x):
        # Custom logic: skip connection and conditional
        h = torch.tanh(self.fc1(x))
        if x.sum() > 0:
            h = h + x[:, :10]  # skip connection (if input large enough)
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
            nn.Linear(8, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class SimpleLSTM(nn.Module):
    """Simple LSTM for testing sequential models."""
    
    def __init__(self, input_size=10, hidden_size=16, num_layers=2, output_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use the last output of the sequence
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


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
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x) 