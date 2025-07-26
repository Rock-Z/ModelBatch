"""
ModelBatch: Train hundreds to thousands of independent PyTorch models simultaneously
on a single GPU using vectorized operations.
"""

__version__ = "0.1.0"

from .callbacks import CallbackPack, Callback
from .core import ModelBatch
from .data import DataRouter
from .optimizer import OptimizerFactory
from .logger import (
    get_logger, 
    configure_logging, 
    set_log_level, 
    add_file_handler,
    get_core_logger,
    get_optuna_logger, 
    get_training_logger
)

__all__ = [
    "CallbackPack",
    "Callback", 
    "DataRouter",
    "ModelBatch",
    "OptimizerFactory",
    "get_logger", 
    "configure_logging", 
    "set_log_level", 
    "add_file_handler",
    "get_core_logger",
    "get_optuna_logger", 
    "get_training_logger"
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