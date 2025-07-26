"""
ModelBatch: Train hundreds to thousands of independent PyTorch models simultaneously
on a single GPU using vectorized operations.
"""

__version__ = "0.1.0"

from .callbacks import CallbackPack
from .core import ModelBatch
from .data import DataRouter
from .optimizer import OptimizerFactory

__all__ = [
    "CallbackPack",
    "DataRouter",
    "ModelBatch",
    "OptimizerFactory",
]

# Optional integrations (only available if dependencies are installed)
try:
    from .optuna_integration import ConstraintSpec, ModelBatchStudy

    __all__ += ["ConstraintSpec", "ModelBatchStudy"]
except ImportError:
    pass

try:
    from .huggingface_integration import (
        HFModelBatch,
        HFTrainerMixin,
        ModelBatchTrainer,
    )

    __all__ += [
        "HFModelBatch",
        "HFTrainerMixin",
        "ModelBatchTrainer",
    ]
except ImportError:
    pass
