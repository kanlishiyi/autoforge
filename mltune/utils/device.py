"""Device utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device.

    Args:
        device: Device string ("cuda", "cpu", "mps", or None for auto)

    Returns:
        torch.device object
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> Tuple[str, int, str]:
    """
    Get device information.

    Returns:
        Tuple of (device_name, memory_gb, device_type)
    """
    device = get_device()

    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return name, memory, "GPU"
    elif device.type == "mps":
        return "Apple Silicon", 0, "GPU"
    else:
        import platform
        return platform.processor() or "CPU", 0, "CPU"


def set_device(device: str) -> None:
    """
    Set the default device.

    Args:
        device: Device string
    """
    torch.set_default_device(device)


def get_num_devices() -> int:
    """
    Get the number of available devices.

    Returns:
        Number of GPUs or 1 for CPU
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def clear_cuda_memory() -> None:
    """Clear CUDA memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage() -> Tuple[float, float]:
    """
    Get current memory usage.

    Returns:
        Tuple of (used_gb, total_gb)
    """
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return used, total
    return 0.0, 0.0
