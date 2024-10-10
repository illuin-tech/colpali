from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import rearrange
from PIL import Image

EPSILON = 1e-10


def get_similarity_maps_from_embeddings(
    image_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    n_patches: Tuple[int, int],
) -> torch.Tensor:
    """
    Get the similarity maps between the query embeddings and the image embeddings.

    Args:
        image_embeddings: tensor of shape (batch_size, n_patches_x * n_patches_y, dim)
        query_embeddings: tensor of shape (batch_size, query_tokens, dim)
        n_patches: tuple of integers representing the number of patches along the x and y dimensions
    """

    # Rearrange the output image tensor to explicitly represent the 2D grid of patches
    image_embedding_grid = rearrange(
        image_embeddings, "b (h w) c -> b h w c", h=n_patches[0], w=n_patches[1]
    )  # (1, n_patches_x, n_patches_y, dim)

    similarity_maps = torch.einsum(
        "bnk,bijk->bnij", query_embeddings, image_embedding_grid
    )  # (1, query_tokens, n_patches_x, n_patches_y)

    return similarity_maps


def normalize_similarity_map(similarity_map: torch.Tensor) -> torch.Tensor:
    """
    Normalize the similarity map to have values in the range [0, 1].

    Args:
        similarity_map: tensor of shape (n_patch_x, n_patch_y) or (batch_size, n_patch_x, n_patch_y)
    """
    if similarity_map.ndim not in [2, 3]:
        raise ValueError(
            "The input tensor must have 2 dimensions (n_patch_x, n_patch_y) or "
            "3 dimensions (batch_size, n_patch_x, n_patch_y)."
        )

    # Compute the minimum values along the last two dimensions (n_patch_x, n_patch_y)
    min_vals = similarity_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]

    # Compute the maximum values along the last two dimensions (n_patch_x, n_patch_y)
    max_vals = similarity_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    # Normalize the tensor
    similarity_map_normalized = (similarity_map - min_vals) / (
        max_vals - min_vals + EPSILON
    )  # NOTE: add a small epsilon to avoid division by zero

    return similarity_map_normalized


def plot_similarity_heatmap(
    image: Image.Image,
    patch_size: int,
    resolution: int,
    similarity_map: torch.Tensor,
    figsize: Tuple[int, int] = (8, 8),
    style: Optional[Union[Dict[str, Any], str]] = None,
    show_colorbar: bool = False,
    show_axes: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap of the similarity map over the image.
    """

    # Resize the image
    image_resized = image.resize((resolution, resolution))

    # Convert the image to an array
    # NOTE: The last channel is the alpha channel in the RGBA format.
    img_array = np.array(image_resized.convert("RGBA"))  # (height, width, channels)

    # Get the number of patches
    if resolution % patch_size != 0:
        raise ValueError("The image resolution must be divisible by the patch size.")

    # Normalize the similarity map
    similarity_map_array = (
        normalize_similarity_map(similarity_map).to(torch.float32).cpu().numpy()
    )  # (n_patches_x, n_patches_y)

    # Default style
    if style is None:
        style = {}

    # Get the attention map as a numpy array
    attention_map_image = Image.fromarray((similarity_map_array * 255).astype("uint8")).resize(
        image_resized.size, Image.Resampling.BICUBIC
    )

    # Create a figure
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)

        # First, plot the image
        ax.imshow(img_array)

        # Then, plot the attention map
        im = ax.imshow(
            attention_map_image,
            cmap=sns.color_palette("mako", as_cmap=True),
            alpha=0.5,
        )

        if show_colorbar:
            fig.colorbar(im)
        if not show_axes:
            ax.set_axis_off()

        fig.tight_layout()

    return fig, ax
