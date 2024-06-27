import logging

import torch

logger = logging.getLogger(__name__)

EPSILON = 1e-10


def normalize_attention_map_per_query_token(x: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the attention map for ColPali for each query token.
    The output tensor will have values in the range [0, 1] and the
    same shape as the input tensor.

    Args:
        x: The attention map tensor of shape (batch_size, n_text_tokens, n_patch_x, n_patch_y).
    """
    if x.ndim != 4:
        raise ValueError("The input tensor must have 4 dimensions.")

    # Compute the minimum values along the last two dimensions (n_patch_x, n_patch_y)
    min_vals = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]

    # Compute the maximum values along the last two dimensions (n_patch_x, n_patch_y)
    max_vals = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    # Normalize the tensor
    x_normalized = (x - min_vals) / (max_vals - min_vals + EPSILON)  # Adding a small epsilon to avoid division by zero

    return x_normalized


def normalize_attention_map_per_query(x: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the attention map for ColPali for each query token.
    The output tensor will have values in the range [0, 1] and the
    same shape as the input tensor.

    Args:
        x: The attention map tensor of shape (batch_size, n_text_tokens, n_patch_x, n_patch_y).
    """
    # Log warning
    logger.warning(
        "This function should not be used for ColPali because it doesn't make sense to normalize the attention map across the text tokens."
    )

    if x.ndim != 4:
        raise ValueError("The input tensor must have 4 dimensions.")

    # Compute the minimum values along the last three dimensions (n_text_tokens, n_patch_x, n_patch_y)
    min_vals = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0].min(dim=-3, keepdim=True)[0]

    # Compute the maximum values along the last three dimensions (n_text_tokens, n_patch_x, n_patch_y)
    max_vals = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=-3, keepdim=True)[0]

    # Normalize the tensor
    x_normalized = (x - min_vals) / (max_vals - min_vals + EPSILON)  # Adding a small epsilon to avoid division by zero

    return x_normalized
