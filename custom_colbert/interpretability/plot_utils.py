from typing import Any, Dict, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch
from PIL import Image

MAX_OPACITY = 255


def plot_patches(
    img: Image.Image,
    patch_size: int,
    image_resolution: int,
    patch_opacities: Optional[npt.NDArray | torch.Tensor] = None,
    figsize: Tuple[int, int] = (8, 8),
    style: Dict[str, Any] | str | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot patches of a square image.
    Set `style` to "dark_background" if your image has a light background.
    """

    # Get the number of patches
    if image_resolution % patch_size != 0:
        raise ValueError("The image resolution must be divisible by the patch size.")
    num_patches = image_resolution // patch_size

    # Default style
    if style is None:
        style = {}

    # Sanity checks
    if patch_opacities is not None:
        if isinstance(patch_opacities, torch.Tensor):
            patch_opacities = cast(npt.NDArray, patch_opacities.cpu().numpy())
        if patch_opacities.shape != (num_patches, num_patches):
            raise ValueError("The shape of the patch_opacities tensor is not correct.")
        if not np.all((0 <= patch_opacities) & (patch_opacities <= 1)):
            raise ValueError("The patch_opacities tensor must have values between 0 and 1.")

    # If the image is not square, raise an error
    if img.size[0] != img.size[1]:
        raise ValueError("The image must be square.")

    # Get the image as a numpy array
    img_array = np.array(img.convert("RGBA"))  # (H, W, C) where the last channel is the alpha channel

    # Create a figure
    with plt.style.context(style):
        fig, axis = plt.subplots(num_patches, num_patches, figsize=figsize)

        # Plot the patches
        for i in range(num_patches):
            for j in range(num_patches):
                patch = img_array[i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size, :]
                # Set the opacity of the patch
                if patch_opacities is not None:
                    patch[:, :, -1] = round(patch_opacities[i, j] * MAX_OPACITY)
                axis[i, j].imshow(patch)
                axis[i, j].axis("off")

        fig.subplots_adjust(wspace=0.1, hspace=0.1)

    fig.tight_layout()

    return fig, axis


def plot_attention_heatmap(
    img: Image.Image,
    patch_size: int,
    image_resolution: int,
    attention_map: npt.NDArray | torch.Tensor,
    figsize: Tuple[int, int] = (8, 8),
    style: Dict[str, Any] | str | None = None,
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
    if isinstance(attention_map, torch.Tensor):
        attention_map = cast(npt.NDArray, attention_map.cpu().numpy())
    if attention_map.shape != (num_patches, num_patches):
        raise ValueError("The shape of the patch_opacities tensor is not correct.")
    if not np.all((0 <= attention_map) & (attention_map <= 1)):
        raise ValueError("The patch_opacities tensor must have values between 0 and 1.")

    # If the image is not square, raise an error
    if img.size[0] != img.size[1]:
        raise ValueError("The image must be square.")

    # Get the image as a numpy array
    img_array = np.array(img.convert("RGBA"))  # (H, W, C) where the last channel is the alpha channel

    # Get the attention map as a numpy array
    attention_map_image = Image.fromarray((attention_map * 255).astype("uint8")).resize(
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
