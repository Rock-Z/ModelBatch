"""
ModelBatch: Train many independent PyTorch models simultaneously on a single GPU
using vectorized operations.
"""

__version__ = "0.1.0"

from .callbacks import Callback, CallbackPack
from .core import ModelBatch
from .data import DataRouter
from .logger import (
    add_file_handler,
    configure_logging,
    get_core_logger,
    get_logger,
    get_optuna_logger,
    get_training_logger,
    set_log_level,
)
from .optimizer import OptimizerFactory

__all__ = [
    "Callback",
    "CallbackPack",
    "DataRouter",
    "ModelBatch",
    "OptimizerFactory",
    "add_file_handler",
    "configure_logging",
    "get_core_logger",
    "get_logger",
    "get_optuna_logger",
    "get_training_logger",
    "set_log_level"
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
