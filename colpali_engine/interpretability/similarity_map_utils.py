from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange

EPSILON = 1e-10


def get_similarity_maps_from_embeddings(
    image_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    n_patches: Union[Tuple[int, int], List[Tuple[int, int]]],
    image_mask: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Get the batched similarity maps between the query embeddings and the image embeddings.
    Each element in the returned list is a tensor of shape (query_tokens, n_patches_x, n_patches_y).

    Args:
        image_embeddings: tensor of shape (batch_size, image_tokens, dim)
        query_embeddings: tensor of shape (batch_size, query_tokens, dim)
        n_patches: number of patches per dimension for each image in the batch. If a single tuple is provided,
            the same number of patches is used for all images in the batch (broadcasted).
        image_mask: tensor of shape (batch_size, image_tokens). Used to filter out the embeddings
            that are not related to the image
    """

    if isinstance(n_patches, tuple):
        n_patches = [n_patches] * image_embeddings.size(0)

    similarity_maps: List[torch.Tensor] = []

    for idx in range(image_embeddings.size(0)):
        # Sanity check
        if image_mask[idx].sum() != n_patches[idx][0] * n_patches[idx][1]:
            raise ValueError(
                f"The number of patches ({n_patches[idx][0]} x {n_patches[idx][1]} = "
                f"{n_patches[idx][0] * n_patches[idx][1]}) "
                f"does not match the number of non-padded image tokens ({image_mask[idx].sum()})."
            )

        # Rearrange the output image tensor to explicitly represent the 2D grid of patches
        image_embedding_grid = rearrange(
            image_embeddings[idx][image_mask[idx]],  # (n_patches_x * n_patches_y, dim)
            "(h w) c -> w h c",
            w=n_patches[idx][0],
            h=n_patches[idx][1],
        )  # (n_patches_x, n_patches_y, dim)

        similarity_map = torch.einsum(
            "nk,ijk->nij", query_embeddings[idx], image_embedding_grid
        )  # (batch_size, query_tokens, n_patches_x, n_patches_y)

        similarity_maps.append(similarity_map)

    return similarity_maps


def normalize_similarity_map(
    similarity_map: torch.Tensor,
    value_range: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """
    Normalize the similarity map to have values in the range [0, 1].

    Args:
        similarity_map: tensor of shape (n_patch_x, n_patch_y) or (batch_size, n_patch_x, n_patch_y)
        value_range: optional tuple specifying the (min, max) range to use for normalization.
            When None, the min/max are computed from the input tensor (default behavior).
    """
    if similarity_map.ndim not in [2, 3]:
        raise ValueError(
            "The input tensor must have 2 dimensions (n_patch_x, n_patch_y) or "
            "3 dimensions (batch_size, n_patch_x, n_patch_y)."
        )

    if value_range is None:
        # Compute the minimum values along the last two dimensions (n_patch_x, n_patch_y)
        min_vals = similarity_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[
            0
        ]  # (1, 1) or (batch_size, 1, 1)

        # Compute the maximum values along the last two dimensions (n_patch_x, n_patch_y)
        max_vals = similarity_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[
            0
        ]  # (1, 1) or (batch_size, 1, 1)
    else:
        min_vals, max_vals = value_range
        broadcast_shape = (1,) * similarity_map.ndim
        min_vals = torch.as_tensor(min_vals, dtype=similarity_map.dtype, device=similarity_map.device).view(
            broadcast_shape
        )
        max_vals = torch.as_tensor(max_vals, dtype=similarity_map.dtype, device=similarity_map.device).view(
            broadcast_shape
        )

    # Normalize the tensor
    # NOTE: Add a small epsilon to avoid division by zero.
    similarity_map_normalized = (similarity_map - min_vals) / (
        max_vals - min_vals + EPSILON
    )  # (n_patch_x, n_patch_y) or (batch_size, n_patch_x, n_patch_y)

    return similarity_map_normalized
