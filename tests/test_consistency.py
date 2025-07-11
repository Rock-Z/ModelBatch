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
from test_models import SimpleMLP, CustomModel, DeepMLP, SimpleLSTM, SimpleCNN

def assert_allclose_tensor(tensor1, tensor2, rtol=1e-5, atol=1e-8):
    """Helper function to assert two tensors are close."""
    assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

class TestModelBatchConsistency:
    """Test class for ModelBatch consistency with individual models."""
    
    # Test configurations for different model types
    MODEL_CONFIGS = [
        # (model_class, model_params, num_models, input_shape, target_shape, batch_size)
        (SimpleMLP, {"input_size": 8, "output_size": 4}, 3, (6, 8), (6,), 6),
        (SimpleMLP, {"input_size": 7, "output_size": 5}, 2, (4, 7), (4,), 4),
        (SimpleMLP, {"input_size": 6, "output_size": 3}, 2, (3, 6), (3,), 3),
        
        (CustomModel, {"input_size": 8, "output_size": 4}, 3, (6, 8), (6,), 6),
        (CustomModel, {"input_size": 7, "output_size": 5}, 2, (4, 7), (4,), 4),
        (CustomModel, {"input_size": 6, "output_size": 3}, 2, (3, 6), (3,), 3),
        
        (DeepMLP, {"input_size": 8, "output_size": 4}, 3, (6, 8), (6,), 6),
        (DeepMLP, {"input_size": 7, "output_size": 5}, 2, (4, 7), (4,), 4),
        (DeepMLP, {"input_size": 6, "output_size": 3}, 2, (3, 6), (3,), 3),
        
        (SimpleLSTM, {"input_size": 8, "hidden_size": 12, "output_size": 4}, 3, (6, 10, 8), (6,), 6),
        (SimpleLSTM, {"input_size": 7, "hidden_size": 10, "output_size": 5}, 2, (4, 8, 7), (4,), 4),
        (SimpleLSTM, {"input_size": 6, "hidden_size": 8, "output_size": 3}, 2, (3, 6, 6), (3,), 3),
        
        (SimpleCNN, {"input_channels": 1, "num_classes": 4}, 3, (6, 1, 32, 32), (6,), 6),
        (SimpleCNN, {"input_channels": 3, "num_classes": 5}, 2, (4, 3, 32, 32), (4,), 4),
        (SimpleCNN, {"input_channels": 1, "num_classes": 3}, 2, (3, 1, 32, 32), (3,), 3),
    ]
    
    @pytest.mark.parametrize("model_class,model_params,num_models,input_shape,target_shape,batch_size", MODEL_CONFIGS)
    def test_output_consistency(self, model_class, model_params, num_models, input_shape, target_shape, batch_size):
        """Test that ModelBatch outputs are consistent with individual model outputs."""
        models = create_identical_models(model_class, model_params, num_models)
        mb_shared = ModelBatch(models, shared_input=True)
        mb_nonshared = ModelBatch(models, shared_input=False)
        
        # Create input tensor
        input_tensor = torch.randn(input_shape)
        
        # Test shared input
        mb_out = mb_shared(input_tensor)
        for i, model in enumerate(models):
            ref_out = model(input_tensor)
            assert torch.allclose(mb_out[i], ref_out)
        
        # Test non-shared input
        input_tensor_ns = torch.randn(num_models, *input_shape)
        mb_out_ns = mb_nonshared(input_tensor_ns)
        for i, model in enumerate(models):
            ref_out = model(input_tensor_ns[i])
            assert torch.allclose(mb_out_ns[i], ref_out)
    
    @pytest.mark.parametrize("model_class,model_params,num_models,input_shape,target_shape,batch_size", MODEL_CONFIGS)
    def test_loss_consistency(self, model_class, model_params, num_models, input_shape, target_shape, batch_size):
        """Test that ModelBatch loss computation is consistent with individual model losses."""
        models = create_identical_models(model_class, model_params, num_models)
        mb = ModelBatch(models)
        
        # Create input and targets
        input_tensor = torch.randn(input_shape)
        num_classes = model_params.get("output_size", model_params.get("num_classes", 5))
        targets = torch.randint(0, num_classes, target_shape)
        
        # Compute outputs and loss
        outputs = mb(input_tensor)
        loss_fn = F.cross_entropy
        mb_loss = mb.compute_loss(outputs, targets, loss_fn)
        
        # Compute individual losses
        ref_losses = []
        for i, model in enumerate(models):
            out = model(input_tensor)
            ref_losses.append(loss_fn(out, targets))
        ref_losses = torch.stack(ref_losses)
        
        # Assertions
        assert mb.latest_losses is not None
        assert torch.allclose(mb.latest_losses, ref_losses)
        assert torch.isclose(mb_loss, ref_losses.mean())
    
    @pytest.mark.parametrize("model_class,model_params,num_models,input_shape,target_shape,batch_size", MODEL_CONFIGS)
    def test_gradient_consistency(self, model_class, model_params, num_models, input_shape, target_shape, batch_size):
        """Test that ModelBatch gradients are consistent with individual model gradients."""
        models = create_identical_models(model_class, model_params, num_models)
        mb = ModelBatch(models)
        
        # Create input and targets
        input_tensor = torch.randn(input_shape, requires_grad=True)
        num_classes = model_params.get("output_size", model_params.get("num_classes", 5))
        targets = torch.randint(0, num_classes, target_shape)
        
        # Compute outputs and loss, then backward
        outputs = mb(input_tensor)
        loss = mb.compute_loss(outputs, targets, F.cross_entropy)
        loss.backward()
        
        # Compute individual gradients
        for i, model in enumerate(models):
            model.zero_grad()
            out = model(input_tensor)
            l = F.cross_entropy(out, targets)
            l.backward()
        
        # Compare gradients directly
        for i, model in enumerate(models):
            for p_mb, p_ind in zip(mb.models[i].parameters(), model.parameters()):
                g_mb = p_mb.grad
                g_ind = p_ind.grad
                if g_mb is not None and g_ind is not None:
                    assert isinstance(g_mb, torch.Tensor) and isinstance(g_ind, torch.Tensor)
                    assert_allclose_tensor(g_mb, g_ind)
    
    def test_specific_model_combinations(self):
        """Test specific combinations that might have edge cases."""
        # Test with different model counts
        for num_models in [1, 2, 5]:
            models = create_identical_models(SimpleMLP, {"input_size": 4, "output_size": 2}, num_models)
            mb = ModelBatch(models)
            input_tensor = torch.randn(3, 4)
            outputs = mb(input_tensor)
            assert outputs.shape[0] == num_models
        
        # Test with very small inputs
        models = create_identical_models(SimpleMLP, {"input_size": 2, "output_size": 1}, 2)
        mb = ModelBatch(models)
        input_tensor = torch.randn(1, 2)
        outputs = mb(input_tensor)
        assert outputs.shape == (2, 1, 1)
    
    def test_empty_model_list(self):
        """Test that ModelBatch raises ValueError with empty model list."""
        with pytest.raises(ValueError):
            ModelBatch([])
    
    def test_mismatched_model_types(self):
        """Test that ModelBatch works with different model types."""
        models = [SimpleMLP(input_size=4, output_size=2), CustomModel(input_size=4, output_size=2)]
        with pytest.raises(ValueError):
            ModelBatch(models)