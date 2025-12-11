#!/usr/bin/env python3
"""
Simplified example for generating ColGemma3 interpretability maps.

This example follows the same user-friendly API pattern as the ColModernVBert example,
using ColGemma3 (ColNetraEmbed) for multi-vector late interaction retrieval with
attention heatmap visualization.

Usage:
    python examples/interpretability/colgemma/generate_interpretability_maps.py
"""

from pathlib import Path
import uuid
from typing import cast, Any
import math

import matplotlib.pyplot as plt
import torch
from PIL import Image

from colpali_engine.interpretability import get_similarity_maps_from_embeddings
from colpali_engine.interpretability.similarity_maps import plot_all_similarity_maps
from colpali_engine.models import ColGemma3, ColGemmaProcessor3


def main():
    print("=== ColGemma3 (ColNetraEmbed) Interpretability Example ===\n")

    # ==================== USER INPUTS ====================
    use_real_document = True  # Set to False to use a blank test image
    # =====================================================

    if use_real_document:
        # Load a real document from DocVQA dataset
        print("Loading a real document from DocVQA dataset...")
        from datasets import load_dataset

        dataset = load_dataset(
            "vidore/docvqa_test_subsampled", split="test", streaming=True
        )
        # streaming datasets may yield values that type checkers treat as Sequence;
        # cast to dict so string indexing (sample["image"]) is accepted by the type checker.
        sample = dict(next(iter(dataset)))
        image = sample["image"]
        query = sample["query"]
        print(f"Document loaded! Query: '{query}'")
    else:
        # For demo purposes, use a simple test image
        print("Creating a demo test image...")
        image = Image.new("RGB", (800, 600), color="white")
        query = "What is the total revenue?"

    # Load model and processor
    print("Loading ColGemma3 model and processor...")
    print("Note: First load may take a few minutes to download the model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = ColGemmaProcessor3.from_pretrained(
        "Cognitive-Lab/ColNetraEmbed",
        use_fast=True,
    )
    model = ColGemma3.from_pretrained(
        "Cognitive-Lab/ColNetraEmbed",
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()
    print("Model loaded successfully!")

    # Preprocess inputs
    print(f"\nProcessing query: '{query}'")
    batch_images = processor.process_images([image]).to(device)
    batch_queries = processor.process_queries([query]).to(device)

    # Forward passes
    print("Computing embeddings...")
    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Query embeddings shape: {query_embeddings.shape}")

    # Create image mask manually (ColGemmaProcessor3 doesn't have get_image_mask)
    print("\nCreating image mask...")
    if "input_ids" in batch_images and hasattr(model.config, "image_token_id"):
        image_token_id = model.config.image_token_id
        image_mask = batch_images["input_ids"] == image_token_id
        print(f"Image mask shape: {image_mask.shape}")
        print(f"Number of image tokens: {image_mask.sum().item()}")
    else:
        # Fallback: assume all tokens are image tokens
        image_mask = torch.ones(
            image_embeddings.shape[0], image_embeddings.shape[1],
            dtype=torch.bool, device=device
        )
        print(f"Using all-ones mask: {image_mask.shape}")

    # Calculate n_patches from actual number of image tokens
    num_image_tokens = image_mask.sum().item()
    print(f"\nCalculating patch configuration...")
    print(f"Number of image tokens: {num_image_tokens}")

    n_side = int(math.sqrt(num_image_tokens))
    if n_side * n_side == num_image_tokens:
        n_patches = (n_side, n_side)
        print(f"Number of patches: {n_patches[0]} x {n_patches[1]}")
    else:
        print(f"Warning: Image tokens ({num_image_tokens}) is not a perfect square!")
        # Fallback: use default calculation
        n_patches = (16, 16)
        print(f"Using default n_patches: {n_patches}")

    # Generate similarity maps
    print("\nGenerating similarity maps...")
    similarity_maps_batch = get_similarity_maps_from_embeddings(
        image_embeddings=image_embeddings,
        query_embeddings=query_embeddings,
        n_patches=n_patches,
        image_mask=image_mask,
    )

    # Get the similarity map for our input image
    similarity_maps = similarity_maps_batch[0]  # (query_length, n_patches_x, n_patches_y)
    print(f"Similarity map shape: {similarity_maps.shape}")

    # Get query tokens (filtering out special tokens)
    input_ids = batch_queries.input_ids[0].tolist()
    query_tokens = processor.tokenizer.convert_ids_to_tokens(batch_queries.input_ids[0])
    special_token_ids = set(processor.tokenizer.all_special_ids or [])

    filtered_tokens = []
    filtered_indices = []
    for idx, (token, token_id) in enumerate(zip(query_tokens, input_ids)):
        if token_id in special_token_ids:
            continue
        filtered_tokens.append(token)
        filtered_indices.append(idx)

    # Filter similarity maps to match tokens
    similarity_maps = similarity_maps[filtered_indices]

    # Convert to float32 if needed for visualization
    if similarity_maps.dtype == torch.bfloat16:
        similarity_maps = similarity_maps.float()

    # Clean tokens for display (remove special characters that may cause encoding issues)
    display_tokens = [t.replace("Ġ", " ").replace("▁", " ") for t in filtered_tokens]
    print(f"\nQuery tokens: {display_tokens}")
    print(
        f"Similarity range: [{similarity_maps.min().item():.3f}, {similarity_maps.max().item():.3f}]"
    )

    # Generate all similarity maps
    print("\nGenerating similarity map visualizations for all tokens...")
    plots = plot_all_similarity_maps(
        image=image,
        query_tokens=filtered_tokens,
        similarity_maps=similarity_maps,
        figsize=(8, 8),
        show_colorbar=True,
        add_title=True,
    )

    # Save the plots
    output_dir = Path("outputs/interpretability/colgemma/" + uuid.uuid4().hex[:8])
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (fig, ax) in enumerate(plots):
        token = filtered_tokens[idx]
        # Sanitize token for filename (remove special characters)
        token_safe = (
            token.replace("<", "")
            .replace(">", "")
            .replace("Ġ", "")
            .replace("▁", "")
            .replace("?", "")
            .replace(":", "")
            .replace("/", "")
            .replace("\\", "")
            .replace("|", "")
            .replace("*", "")
            .replace('"', "")
        )
        if not token_safe:
            token_safe = f"token_{idx}"
        savepath = output_dir / f"similarity_map_{idx}_{token_safe}.png"
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
        print(f"  Saved: {savepath.name}")
        plt.close(fig)

    print(f"\n[SUCCESS] All similarity maps saved to: {output_dir.absolute()}")

    # Also create an aggregated heatmap (mean across all query tokens)
    print("\nGenerating aggregated heatmap (mean across all query tokens)...")
    aggregated_map = torch.mean(similarity_maps, dim=0)

    from colpali_engine.interpretability.similarity_map_utils import normalize_similarity_map
    import seaborn as sns
    import numpy as np
    from einops import rearrange
    import io

    # Convert the image to an array
    img_array = np.array(image.convert("RGBA"))

    # Normalize the similarity map and convert to numpy
    similarity_map_array = normalize_similarity_map(aggregated_map).to(torch.float32).cpu().numpy()

    # Reshape to match PIL convention
    similarity_map_array = rearrange(similarity_map_array, "h w -> w h")

    # Create PIL image from similarity map
    similarity_map_image = Image.fromarray((similarity_map_array * 255).astype("uint8")).resize(
        image.size, Image.Resampling.BICUBIC
    )

    # Create matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].set_axis_off()

    # Heatmap overlay
    axes[1].imshow(img_array)
    axes[1].imshow(
        similarity_map_image,
        cmap=sns.color_palette("mako", as_cmap=True),
        alpha=0.5,
    )
    axes[1].set_title(f'Aggregated Heatmap: "{query}"', fontsize=14, fontweight="bold")
    axes[1].set_axis_off()

    plt.tight_layout()

    # Save the aggregated heatmap
    aggregated_path = output_dir / "aggregated_heatmap.png"
    plt.savefig(aggregated_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {aggregated_path.name}")
    print(f"\n[SUCCESS] Aggregated heatmap saved to: {aggregated_path.absolute()}")


if __name__ == "__main__":
    main()
