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
    from .optuna_integration import ModelBatchStudy, ConstraintSpec
    __all__.extend(["ModelBatchStudy", "ConstraintSpec"])
except ImportError:
    pass

try:
    from .huggingface_integration import HFModelBatchStudy, HFConstraintSpec
    __all__.extend(["HFModelBatchStudy", "HFConstraintSpec"])
except ImportError:
    pass 