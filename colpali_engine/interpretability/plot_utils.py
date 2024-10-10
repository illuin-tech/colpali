from typing import Any, Dict, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch
from PIL import Image


def plot_similarity_heatmap(
    img: Image.Image,
    patch_size: int,
    image_resolution: int,
    similarity_map: Union[npt.NDArray, torch.Tensor],
    figsize: Tuple[int, int] = (8, 8),
    style: Optional[Union[Dict[str, Any], str]] = None,
    show_colorbar: bool = False,
    show_axes: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap of the attention map over the image.
    The image must be square and `attention_map` must be normalized between 0 and 1.
    """

    # Get the number of patches
    if image_resolution % patch_size != 0:
        raise ValueError("The image resolution must be divisible by the patch size.")
    num_patches = image_resolution // patch_size

    # Default style
    if style is None:
        style = {}

    # Sanity checks
    if isinstance(similarity_map, torch.Tensor):
        similarity_map = cast(npt.NDArray, similarity_map.to(torch.float32).cpu().numpy())
    if similarity_map.shape != (num_patches, num_patches):
        raise ValueError("The shape of the patch_opacities tensor is not correct.")
    if not np.all((0 <= similarity_map) & (similarity_map <= 1)):
        raise ValueError("The patch_opacities tensor must have values between 0 and 1.")

    # If the image is not square, raise an error
    if img.size[0] != img.size[1]:
        raise ValueError("The image must be square.")

    # Get the image as a numpy array
    img_array = np.array(img.convert("RGBA"))  # (H, W, C) where the last channel is the alpha channel

    # Get the attention map as a numpy array
    attention_map_image = Image.fromarray((similarity_map * 255).astype("uint8")).resize(
        img.size, Image.Resampling.BICUBIC
    )

    # Create a figure
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img_array)
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
