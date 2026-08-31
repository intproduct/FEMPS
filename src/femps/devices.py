"""Device selection helpers shared by research scripts."""

from __future__ import annotations

import torch


def resolve_device(specification: str) -> torch.device:
    """Resolve a torch device, with ``auto`` choosing the highest CUDA CC.

    CUDA enumeration is not assumed to match ``nvidia-smi`` output order. This
    matters on the development workstation, where two unsupported V100 cards
    precede the Blackwell card in PyTorch's enumeration.
    """
    if specification != "auto":
        return torch.device(specification)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_index = max(
        range(torch.cuda.device_count()),
        key=lambda index: torch.cuda.get_device_capability(index),
    )
    return torch.device(f"cuda:{best_index}")

