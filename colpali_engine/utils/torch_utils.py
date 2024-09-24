import gc
import logging
from typing import List, TypeVar

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
T = TypeVar("T")


def get_torch_device(device: str = "auto") -> str:
    """
    Returns the device (string) to be used by PyTorch.

    `device` arg defaults to "auto" which will use:
    - "cuda:0" if available
    - else "mps" if available
    - else "cpu".
    """

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():  # for Apple Silicon
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")

    return device


def tear_down_torch():
    """
    Teardown for PyTorch.
    Clears GPU cache for both CUDA and MPS.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.mps.empty_cache()


class ListDataset(Dataset[T]):
    def __init__(self, elements: List[T]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> T:
        return self.elements[idx]
