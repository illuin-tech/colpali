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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class ListDataset(Dataset[T]):
    def __init__(self, elements: List[T]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> T:
        return self.elements[idx]


def unbind_padded_multivector_embeddings(
    embeddings: torch.Tensor,
    padding_value: float = 0.0,
    padding_side: str = "left",
) -> List[torch.Tensor]:
    """
    Removes padding elements from a batch of multivector embeddings.

    Args:
        embeddings (torch.Tensor): A tensor of shape (batch_size, seq_length, dim) with padding.
        padding_value (float): The value used for padding. Each padded token is assumed
            to be a vector where every element equals this value.
        padding_side (str): Either "left" or "right". This indicates whether the padded
            elements appear at the beginning (left) or end (right) of the sequence.

    Returns:
        List[torch.Tensor]: A list of tensors, one per sequence in the batch, where
            each tensor has shape (new_seq_length, dim) and contains only the non-padding elements.
    """
    results: List[torch.Tensor] = []

    for seq in embeddings:
        is_padding = torch.all(seq.eq(padding_value), dim=-1)

        if padding_side == "left":
            non_padding_indices = (~is_padding).nonzero(as_tuple=False)
            if non_padding_indices.numel() == 0:
                valid_seq = seq[:0]
            else:
                first_valid_idx = non_padding_indices[0].item()
                valid_seq = seq[first_valid_idx:]
        elif padding_side == "right":
            non_padding_indices = (~is_padding).nonzero(as_tuple=False)
            if non_padding_indices.numel() == 0:
                valid_seq = seq[:0]
            else:
                last_valid_idx = non_padding_indices[-1].item()
                valid_seq = seq[: last_valid_idx + 1]
        else:
            raise ValueError("padding_side must be either 'left' or 'right'.")
        results.append(valid_seq)

    return results
