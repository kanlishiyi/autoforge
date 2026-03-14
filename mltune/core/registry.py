"""
Registry system for AutoForge components.

Provides a centralized registry for:
- Optimizers
- Models
- Loss functions
- Schedulers
- Custom components
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

T = TypeVar("T")


class Registry:
    """
    Component registry for AutoForge.

    Provides a centralized way to register and retrieve components:

    Example:
        ```python
        # Register a custom optimizer
        @Registry.register_optimizer("my_optimizer")
        class MyOptimizer(BaseOptimizer):
            pass

        # Retrieve optimizer
        optimizer_cls = Registry.get_optimizer("my_optimizer")
        ```
    """

    _optimizers: Dict[str, Type] = {}
    _models: Dict[str, Type] = {}
    _losses: Dict[str, Callable] = {}
    _schedulers: Dict[str, Callable] = {}
    _metrics: Dict[str, Callable] = {}
    _callbacks: Dict[str, Type] = {}
    _loggers: Dict[str, Type] = {}

    @classmethod
    def register_optimizer(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Register an optimizer class.

        Args:
            name: Optimizer name

        Returns:
            Decorator function
        """
        def decorator(optimizer_cls: Type[T]) -> Type[T]:
            cls._optimizers[name] = optimizer_cls
            return optimizer_cls
        return decorator

    @classmethod
    def register_model(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Register a model class."""
        def decorator(model_cls: Type[T]) -> Type[T]:
            cls._models[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def register_loss(cls, name: str) -> Callable[[Callable], Callable]:
        """Register a loss function."""
        def decorator(loss_fn: Callable) -> Callable:
            cls._losses[name] = loss_fn
            return loss_fn
        return decorator

    @classmethod
    def register_scheduler(cls, name: str) -> Callable[[Callable], Callable]:
        """Register a scheduler."""
        def decorator(scheduler_fn: Callable) -> Callable:
            cls._schedulers[name] = scheduler_fn
            return scheduler_fn
        return decorator

    @classmethod
    def register_metric(cls, name: str) -> Callable[[Callable], Callable]:
        """Register a metric function."""
        def decorator(metric_fn: Callable) -> Callable:
            cls._metrics[name] = metric_fn
            return metric_fn
        return decorator

    @classmethod
    def register_callback(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Register a callback class."""
        def decorator(callback_cls: Type[T]) -> Type[T]:
            cls._callbacks[name] = callback_cls
            return callback_cls
        return decorator

    @classmethod
    def register_logger(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Register a logger class."""
        def decorator(logger_cls: Type[T]) -> Type[T]:
            cls._loggers[name] = logger_cls
            return logger_cls
        return decorator

    # Get methods
    @classmethod
    def get_optimizer(cls, name: str) -> Optional[Type]:
        """Get optimizer class by name."""
        return cls._optimizers.get(name)

    @classmethod
    def get_model(cls, name: str) -> Optional[Type]:
        """Get model class by name."""
        return cls._models.get(name)

    @classmethod
    def get_loss(cls, name: str) -> Optional[Callable]:
        """Get loss function by name."""
        return cls._losses.get(name)

    @classmethod
    def get_scheduler(cls, name: str) -> Optional[Callable]:
        """Get scheduler by name."""
        return cls._schedulers.get(name)

    @classmethod
    def get_metric(cls, name: str) -> Optional[Callable]:
        """Get metric function by name."""
        return cls._metrics.get(name)

    @classmethod
    def get_callback(cls, name: str) -> Optional[Type]:
        """Get callback class by name."""
        return cls._callbacks.get(name)

    @classmethod
    def get_logger(cls, name: str) -> Optional[Type]:
        """Get logger class by name."""
        return cls._loggers.get(name)

    # List methods
    @classmethod
    def list_optimizers(cls) -> List[str]:
        """List all registered optimizers."""
        return list(cls._optimizers.keys())

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models."""
        return list(cls._models.keys())

    @classmethod
    def list_losses(cls) -> List[str]:
        """List all registered losses."""
        return list(cls._losses.keys())

    @classmethod
    def list_schedulers(cls) -> List[str]:
        """List all registered schedulers."""
        return list(cls._schedulers.keys())

    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all registered metrics."""
        return list(cls._metrics.keys())

    @classmethod
    def list_callbacks(cls) -> List[str]:
        """List all registered callbacks."""
        return list(cls._callbacks.keys())

    @classmethod
    def list_loggers(cls) -> List[str]:
        """List all registered loggers."""
        return list(cls._loggers.keys())

    # Create instances
    @classmethod
    def create_optimizer(cls, name: str, *args: Any, **kwargs: Any) -> Any:
        """Create optimizer instance."""
        optimizer_cls = cls.get_optimizer(name)
        if optimizer_cls is None:
            raise ValueError(f"Optimizer '{name}' not found. Available: {cls.list_optimizers()}")
        return optimizer_cls(*args, **kwargs)

    @classmethod
    def create_model(cls, name: str, *args: Any, **kwargs: Any) -> Any:
        """Create model instance."""
        model_cls = cls.get_model(name)
        if model_cls is None:
            raise ValueError(f"Model '{name}' not found. Available: {cls.list_models()}")
        return model_cls(*args, **kwargs)

    @classmethod
    def create_logger(cls, name: str, *args: Any, **kwargs: Any) -> Any:
        """Create logger instance."""
        logger_cls = cls.get_logger(name)
        if logger_cls is None:
            raise ValueError(f"Logger '{name}' not found. Available: {cls.list_loggers()}")
        return logger_cls(*args, **kwargs)


# Register common PyTorch optimizers
def _register_torch_optimizers():
    """Register standard PyTorch optimizers."""
    try:
        import torch.optim as optim

        Registry._optimizers["sgd"] = optim.SGD
        Registry._optimizers["adam"] = optim.Adam
        Registry._optimizers["adamw"] = optim.AdamW
        Registry._optimizers["adagrad"] = optim.Adagrad
        Registry._optimizers["rmsprop"] = optim.RMSprop
        Registry._optimizers["adamax"] = optim.Adamax
        Registry._optimizers["nadam"] = optim.NAdam
        Registry._optimizers["lbfgs"] = optim.LBFGS
    except ImportError:
        pass


# Register common loss functions
def _register_torch_losses():
    """Register standard PyTorch loss functions."""
    try:
        import torch.nn as nn

        Registry._losses["mse"] = nn.MSELoss
        Registry._losses["mae"] = nn.L1Loss
        Registry._losses["cross_entropy"] = nn.CrossEntropyLoss
        Registry._losses["bce"] = nn.BCELoss
        Registry._losses["bce_with_logits"] = nn.BCEWithLogitsLoss
        Registry._losses["nll"] = nn.NLLLoss
        Registry._losses["kl_div"] = nn.KLDivLoss
        Registry._losses["huber"] = nn.HuberLoss
        Registry._losses["smooth_l1"] = nn.SmoothL1Loss
        Registry._losses["cosine_embedding"] = nn.CosineEmbeddingLoss
        Registry._losses["margin_ranking"] = nn.MarginRankingLoss
        Registry._losses["triplet_margin"] = nn.TripletMarginLoss
    except ImportError:
        pass


# Register common schedulers
def _register_torch_schedulers():
    """Register standard PyTorch schedulers."""
    try:
        import torch.optim.lr_scheduler as sched

        Registry._schedulers["step"] = sched.StepLR
        Registry._schedulers["multistep"] = sched.MultiStepLR
        Registry._schedulers["exponential"] = sched.ExponentialLR
        Registry._schedulers["cosine"] = sched.CosineAnnealingLR
        Registry._schedulers["cosine_warmup"] = sched.CosineAnnealingWarmRestarts
        Registry._schedulers["cyclic"] = sched.CyclicLR
        Registry._schedulers["onecycle"] = sched.OneCycleLR
        Registry._schedulers["reduce_on_plateau"] = sched.ReduceLROnPlateau
        Registry._schedulers["linear_warmup"] = sched.LinearLR
    except ImportError:
        pass


# Initialize registry with PyTorch components
_register_torch_optimizers()
_register_torch_losses()
_register_torch_schedulers()
