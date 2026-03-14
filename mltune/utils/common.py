"""Common utility functions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp string.
    
    Args:
        fmt: Date format string
        
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime(fmt)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        path: File path
        
    Returns:
        Parsed JSON data
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    data: Dict[str, Any],
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        path: Output path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(
    d: Dict[str, Any],
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Unflatten a dictionary with dot-separated keys.
    
    Args:
        d: Flattened dictionary
        sep: Separator for keys
        
    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_ref = result
        for part in parts[:-1]:
            d_ref = d_ref.setdefault(part, {})
        d_ref[parts[-1]] = value
    return result


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_bytes(bytes_size: int) -> str:
    """
    Format byte size in human-readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}PB"
