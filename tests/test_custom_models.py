import os
import sys
import torch
import torch.nn.functional as F
from torch import nn
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modelbatch import ModelBatch
from modelbatch.utils import create_identical_models
from test_models import CustomModel, DeepMLP, SimpleLSTM, SimpleCNN

class TestModelBatchCustomModels:
    def test_custom_logic_model_batch(self):
        models = create_identical_models(CustomModel, {"input_size": 12, "output_size": 3}, 2)
        mb = ModelBatch(models)
        input_tensor = torch.randn(4, 12)
        out_mb = mb(input_tensor)
        for i, model in enumerate(models):
            out_ref = model(input_tensor)
            assert torch.allclose(out_mb[i], out_ref)

    def test_deep_model_batch(self):
        models = create_identical_models(DeepMLP, {"input_size": 4, "output_size": 2}, 3)
        mb = ModelBatch(models)
        input_tensor = torch.randn(5, 4)
        out_mb = mb(input_tensor)
        for i, model in enumerate(models):
            out_ref = model(input_tensor)
            assert torch.allclose(out_mb[i], out_ref)

    def test_lstm_model_batch(self):
        """Test ModelBatch with LSTM models."""
        models = create_identical_models(SimpleLSTM, {"input_size": 8, "hidden_size": 12, "output_size": 4}, 2)
        mb = ModelBatch(models)
        batch_size = 5
        seq_len = 10
        input_tensor = torch.randn(batch_size, seq_len, 8)
        out_mb = mb(input_tensor)
        for i, model in enumerate(models):
            out_ref = model(input_tensor)
            assert torch.allclose(out_mb[i], out_ref)

    def test_cnn_model_batch(self):
        """Test ModelBatch with CNN models."""
        models = create_identical_models(SimpleCNN, {"input_channels": 1, "num_classes": 5}, 3)
        mb = ModelBatch(models)
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 32, 32)  # 1 channel, 32x32 images
        out_mb = mb(input_tensor)
        for i, model in enumerate(models):
            out_ref = model(input_tensor)
            assert torch.allclose(out_mb[i], out_ref) 