import gc
import logging

import torch

logger = logging.getLogger(__name__)


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
    Should be used after each torch-based vision retriever.
    """
    gc.collect()
    torch.cuda.empty_cache()
