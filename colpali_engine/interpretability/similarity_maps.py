from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import rearrange
from PIL import Image

from colpali_engine.interpretability.similarity_map_utils import normalize_similarity_map


def plot_similarity_map(
    image: Image.Image,
    similarity_map: torch.Tensor,
    figsize: Tuple[int, int] = (8, 8),
    show_colorbar: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot and overlay a similarity map over the input image.

    A similarity map is a 2D tensor where each element (i, j) represents the similarity score between a chosen query
    token and the associated image patch at position (i, j). Thus, the higher the similarity score, the brighter the
    color of the patch.

    To show the returned similarity map, use:

    ```python
    >>> fig, ax = plot_similarity_map(image, similarity_map)
    >>> fig.show()
    ```

    Args:
        image: PIL image
        similarity_map: tensor of shape (n_patches_x, n_patches_y)
        figsize: size of the figure
        show_colorbar: whether to show a colorbar
    """

    # Convert the image to an array
    img_array = np.array(image.convert("RGBA"))  # (height, width, channels)

    # Normalize the similarity map and convert it to Pillow image
    similarity_map_array = (
        normalize_similarity_map(similarity_map).to(torch.float32).cpu().numpy()
    )  # (n_patches_x, n_patches_y)

    # Reshape the similarity map to match the PIL shape convention
    similarity_map_array = rearrange(similarity_map_array, "h w -> w h")  # (n_patches_y, n_patches_x)

    similarity_map_image = Image.fromarray((similarity_map_array * 255).astype("uint8")).resize(
        image.size, Image.Resampling.BICUBIC
    )

    # Create the figure
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(img_array)
        im = ax.imshow(
            similarity_map_image,
            cmap=sns.color_palette("mako", as_cmap=True),
            alpha=0.5,
        )

        if show_colorbar:
            fig.colorbar(im)
        ax.set_axis_off()
        fig.tight_layout()

    return fig, ax


def plot_all_similarity_maps(
    image: Image.Image,
    query_tokens: List[str],
    similarity_maps: torch.Tensor,
    figsize: Tuple[int, int] = (8, 8),
    show_colorbar: bool = False,
    add_title: bool = True,
) -> List[Tuple[plt.Figure, plt.Axes]]:
    """
    For each token in the query, plot and overlay a similarity map over the input image.

    A similarity map is a 2D tensor where each element (i, j) represents the similarity score between a chosen query
    token and the associated image patch at position (i, j). Thus, the higher the similarity score, the brighter the
    color of the patch.

    Args:
        image: PIL image
        query_tokens: list of query tokens
        similarity_maps: tensor of shape (query_tokens, n_patches_x, n_patches_y)
        figsize: size of the figure
        show_colorbar: whether to show a colorbar
        add_title: whether to add a title with the token and the max similarity score

    Example usage for one query-image pair:

    ```python
    >>> from colpali_engine.interpretability.similarity_map_utils import get_similarity_maps_from_embeddings

    >>> batch_images = processor.process_images([image]).to(device)
    >>> batch_queries = processor.process_queries([query]).to(device)

    >>> with torch.no_grad():
            image_embeddings = model.forward(**batch_images)
            query_embeddings = model.forward(**batch_queries)

    >>> n_patches = processor.get_n_patches(
            image_size=image.size,
            patch_size=model.patch_size
        )
    >>> image_mask = processor.get_image_mask(batch_images)

    >>> batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )
    >>> similarity_maps = batched_similarity_maps[0]  # (query_length, n_patches_x, n_patches_y)

    >>> plots = plot_all_similarity_maps(
            image=image,
            query_tokens=query_tokens,
            similarity_maps=similarity_maps,
        )

    >>> for fig, ax in plots:
            fig.show()
    ```
    """

    plots: List[Tuple[plt.Figure, plt.Axes]] = []

    for idx, token in enumerate(query_tokens):
        fig, ax = plot_similarity_map(
            image=image,
            similarity_map=similarity_maps[idx],
            figsize=figsize,
            show_colorbar=show_colorbar,
        )

        if add_title:
            max_sim_score = similarity_maps[idx].max().item()
            ax.set_title(f"Token #{idx}: `{token}`. MaxSim score: {max_sim_score:.2f}", fontsize=14)

        plots.append((fig, ax))

    return plots
