import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from einops import rearrange
from PIL import Image
from tqdm import trange

from colpali_engine.interpretability.plot_utils import plot_similarity_heatmap
from colpali_engine.interpretability.torch_utils import normalize_similarity_map_per_query_token
from colpali_engine.interpretability.vit_configs import VIT_CONFIG
from colpali_engine.models import ColPali, ColPaliProcessor

OUTPUT_DIR = Path("outputs")


@dataclass
class InterpretabilityInput:
    query: str
    image: Image.Image
    start_idx_token: int
    end_idx_token: int


def gen_and_save_similarity_map_per_token(
    model: ColPali,
    processor: ColPaliProcessor,
    query: str,
    image: Image.Image,
    figsize: Tuple[int, int] = (8, 8),
    add_title: bool = True,
    style: str = "dark_background",
    savedir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Generate and save the similarity maps in the `outputs` directory for each token in the query.
    """

    # Sanity checks
    if model.config.name_or_path not in VIT_CONFIG:
        raise ValueError(f"`{model.config.name_or_path}` is not referenced in the VIT_CONFIG dictionary.")
    vit_config = VIT_CONFIG[model.config.name_or_path]

    # Handle savepath
    if not savedir:
        savedir = OUTPUT_DIR / "interpretability" / str(uuid4())
        print(f"No savepath provided. Results will be saved to: `{savedir}`.")
    elif isinstance(savedir, str):
        savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    # Resize the image to square
    input_image_square = image.resize((vit_config.resolution, vit_config.resolution))

    # Preprocess the inputs
    input_text_processed = processor.process_queries([query]).to(model.device)
    input_image_processed = processor.process_images([image]).to(model.device)

    # Forward passes
    with torch.no_grad():
        output_text = model.forward(**input_text_processed)  # (1, query_tokens, dim)
        output_image = model.forward(**input_image_processed)  # (1, n_patches_x * n_patches_y, dim)

    # Remove the special tokens from the output
    output_image = output_image[:, : processor.image_seq_length, :]  # (1, n_patches_x * n_patches_y, dim)

    # Rearrange the output image tensor to explicitly represent the 2D grid of patches
    output_image = rearrange(
        output_image, "b (h w) c -> b h w c", h=vit_config.n_patch_per_dim, w=vit_config.n_patch_per_dim
    )  # (1, n_patches_x, n_patches_y, dim)

    # Get the similarity map
    similarity_map = torch.einsum(
        "bnk,bijk->bnij", output_text, output_image
    )  # (1, query_tokens, n_patches_x, n_patches_y)

    # Normalize the similarity map
    similarity_map_normalized = normalize_similarity_map_per_query_token(
        similarity_map
    )  # (1, query_tokens, n_patches_x, n_patches_y)

    # Get the list of query tokens
    n_tokens = input_text_processed.input_ids.size(1)
    list_query_tokens = processor.tokenizer.tokenize(processor.decode(input_text_processed.input_ids[0]))

    print("\nText tokens:")
    pprint.pprint({idx: val for idx, val in enumerate(list_query_tokens)})
    print("\n")

    # Placeholder
    max_sim_scores_per_token: Dict[str, float] = {}

    # Iterate over the tokens and plot the similarity maps for each token
    for token_idx in trange(1, n_tokens - 1, desc="Iterating over tokens..."):  # exclude the <bos> and the "\n" tokens
        fig, ax = plot_similarity_heatmap(
            input_image_square,
            vit_config.patch_size,
            vit_config.resolution,
            similarity_map=similarity_map_normalized[0, token_idx, :, :],
            figsize=figsize,
            style=style,
        )
        max_sim_score = similarity_map[0, token_idx, :, :].max().item()
        max_sim_scores_per_token[f"{token_idx}: {list_query_tokens[token_idx]}"] = max_sim_score
        if add_title:
            ax.set_title(
                f"Token #{token_idx}: `{list_query_tokens[token_idx]}`. MaxSim score: {max_sim_score:.2f}",
                fontsize=14,
            )

        savepath = savedir / f"token_{token_idx}.png"
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

        print(f"Saved attention map for token {token_idx} (`{list_query_tokens[token_idx]}`) to `{savepath}`.\n")
        plt.close(fig)

    # Plot and save the max similarity scores per token
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(1 * len(max_sim_scores_per_token), 5))

        ser = pd.Series(max_sim_scores_per_token)
        ser.plot.bar(ax=ax)

        ax.set_xlabel("Token")
        ax.set_ylabel("Score")
        ax.set_title("Max similarity score across all patches", fontsize=14)

        savepath = savedir / "max_sim_scores_per_token.png"
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

        print(f"Saved max similarity scores per token to `{savepath}`.\n")
        plt.close(fig)

    return
