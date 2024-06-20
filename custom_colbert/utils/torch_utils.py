"""
Utility functions for interpretability.
"""

import torch


def get_torch_device() -> str:
    """
    Returns the device and dtype to be used for torch tensors.
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():  # for Apple Silicon
        device = "mps"
    else:
        device = "cpu"
    return device
