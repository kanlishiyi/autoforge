"""Random seed utilities for reproducibility."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Sets seed for:
    - Python random module
    - NumPy
    - PyTorch (if available)
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_random_state() -> dict:
    """
    Get the current random state.
    
    Returns:
        Dictionary containing random states
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    
    try:
        import torch
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state()
    except ImportError:
        pass
    
    return state


def set_random_state(state: dict) -> None:
    """
    Restore random state.
    
    Args:
        state: Dictionary containing random states
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    
    try:
        import torch
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state(state["torch_cuda"])
    except ImportError:
        pass


class RandomState:
    """
    Context manager for temporary random state.
    
    Example:
        ```python
        set_seed(42)
        
        with RandomState():
            # Use different random state temporarily
            x = np.random.rand(10)
        
        # Original state restored
        y = np.random.rand(10)  # Same as if no RandomState was used
        ```
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize context manager.
        
        Args:
            seed: Optional seed for temporary state
        """
        self.seed = seed
        self.original_state = None
    
    def __enter__(self):
        """Save original state and optionally set new seed."""
        self.original_state = get_random_state()
        if self.seed is not None:
            set_seed(self.seed)
        return self
    
    def __exit__(self, *args):
        """Restore original state."""
        set_random_state(self.original_state)
