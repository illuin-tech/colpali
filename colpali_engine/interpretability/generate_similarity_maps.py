import pprint
from pathlib import Path
from typing import Annotated, Dict, List, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import typer
from PIL import Image
from tqdm import trange

from colpali_engine.interpretability.similarity_maps import get_similarity_maps_from_embeddings, plot_similarity_heatmap
from colpali_engine.interpretability.vit_configs import MODEL_NAME_TO_VIT_CONFIG, BaseViTConfig
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device

OUTPUT_DIR = Path("outputs")

app = typer.Typer(
    help="CLI for generating similarity maps for ColPali.",
    no_args_is_help=True,
)


def generate_similarity_maps(
    model: ColPali,
    processor: ColPaliProcessor,
    vit_config: BaseViTConfig,
    query: str,
    image: Image.Image,
    savedir: str,
    figsize: Tuple[int, int] = (8, 8),
    add_title: bool = True,
    style: str = "dark_background",
) -> None:
    """
    High-level function to generate and save the similarity maps for each token in the query.
    """
    # Preprocess the inputs
    batch_images = processor.process_images([image]).to(model.device)
    batch_queries = processor.process_queries([query]).to(model.device)

    # Forward passes
    with torch.no_grad():
        image_embeddings = model.forward(**batch_images)  # (1, n_patches_x * n_patches_y + n_special_tokens, dim)
        query_embeddings = model.forward(**batch_queries)  # (1, query_tokens, dim)

    # Remove the special tokens from the output
    image_embeddings = image_embeddings[:, : processor.image_seq_length, :]  # (1, n_patches_x * n_patches_y, dim)

    # Get the similarity maps
    similarity_maps = get_similarity_maps_from_embeddings(
        image_embeddings,
        query_embeddings,
        n_patches=vit_config.get_n_patches(image=image),
    )

    # Get the list of query tokens
    n_tokens = batch_queries.input_ids.size(1)
    list_query_tokens = processor.tokenizer.tokenize(processor.decode(batch_queries.input_ids[0]))

    print("\nText tokens:")
    pprint.pprint({idx: val for idx, val in enumerate(list_query_tokens)})
    print("\n")

    # Placeholder
    max_sim_scores_per_token: Dict[str, float] = {}

    # Create the output directory if it does not exist
    Path(savedir).mkdir(parents=True, exist_ok=True)

    # Iterate over the tokens and plot the similarity maps for each token
    for token_idx in trange(1, n_tokens - 1, desc="Iterating over tokens..."):  # exclude the <bos> and the "\n" tokens
        fig, ax = plot_similarity_heatmap(
            image=image,
            similarity_map=similarity_maps[0, token_idx, :, :],
            resolution=vit_config.resolution,
            figsize=figsize,
            style=style,
        )

        max_sim_score = similarity_maps[0, token_idx, :, :].max().item()
        max_sim_scores_per_token[f"{token_idx}: {list_query_tokens[token_idx]}"] = max_sim_score

        if add_title:
            ax.set_title(
                f"Token #{token_idx}: `{list_query_tokens[token_idx]}`. MaxSim score: {max_sim_score:.2f}",
                fontsize=14,
            )

        savepath = Path(savedir) / f"token_{token_idx}.png"
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

        savepath = Path(savedir) / "max_sim_scores_per_token.png"
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

        print(f"Saved max similarity scores per token to `{savepath}`.\n")
        plt.close(fig)

    return


@app.command()
def main(
    model_name: Annotated[str, typer.Option(help="Model name for the ColPali model")],
    documents: Annotated[List[Path], typer.Option(help="List of document filepaths (image format)")],
    queries: Annotated[List[str], typer.Option(help="List of queries")],
    device: Annotated[str, typer.Option(help="Torch device")] = "auto",
) -> None:
    """
    Load the ColPali model and, for each query-document pair, generate similarity maps fo
    each token in the current query.
    """

    # Sanity checks
    assert len(documents) == len(queries), "The number of documents and queries must be the same."
    for document in documents:
        if not document.is_file():
            raise FileNotFoundError(f"File not found: `{document}`")

    # Set the device
    device = get_torch_device(device)
    print(f"Using device: {device}")

    # Load the model
    model = cast(
        ColPali,
        ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ),
    ).eval()

    # Load the processor
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))
    print("Loaded custom processor.\n")

    # Get the ViT config
    if model_name not in MODEL_NAME_TO_VIT_CONFIG:
        raise ValueError(f"`{model_name}` is not supported by the CLI tool.")
    vit_config = MODEL_NAME_TO_VIT_CONFIG[model_name]

    # Load the images
    images = [Image.open(img_filepath) for img_filepath in documents]

    # Generate the similarity maps
    for query, image, filepath in zip(queries, images, documents):
        print(f"\n\nProcessing query `{query}` and document `{filepath}`\n")
        savedir = OUTPUT_DIR / "interpretability" / filepath.stem
        generate_similarity_maps(
            model=model,
            processor=processor,
            vit_config=vit_config,
            query=query,
            image=image,
            savedir=str(savedir),
        )

    print("\nDone.")


if __name__ == "__main__":
    app()
