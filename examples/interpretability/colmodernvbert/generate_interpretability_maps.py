#!/usr/bin/env python3
"""
Simplified example for generating ColModernVBert interpretability maps.

This example follows the same user-friendly API pattern as the ColPali cookbook,
but uses ColModernVBert with automatic handling of Idefics3-style image splitting.

Usage:
    python examples/interpretability/colmodernvbert/simple_interpretability_example.py
"""

import uuid
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import torch
from PIL import Image

from colpali_engine.interpretability.similarity_maps import plot_all_similarity_maps
from colpali_engine.models import ColModernVBert, ColModernVBertProcessor


def main():
    print("=== ColModernVBert Simple Interpretability Example ===\n")

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
    print("Loading model and processor...")
    processor = ColModernVBertProcessor.from_pretrained("ModernVBERT/colmodernvbert")
    model = ColModernVBert.from_pretrained("ModernVBERT/colmodernvbert")
    model.eval()

    # Preprocess inputs
    print(f"\nProcessing query: '{query}'")
    batch_images = processor.process_images([image])
    batch_queries = processor.process_queries([query])

    # Forward passes
    print("Computing embeddings...")
    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    # Get the number of image patches
    n_patches = processor.get_n_patches((image.size[1], image.size[0]))
    print(f"Number of patches: {n_patches[0]} x {n_patches[1]}")
    # Get LOCAL image mask (excludes global patch for spatial correspondence)
    image_mask = processor.get_local_image_mask(cast(Any, batch_images))

    # Generate similarity maps using the processor's method
    # This automatically handles Idefics3-style sub-patch rearrangement!
    similarity_maps_batch = processor.get_similarity_maps_from_embeddings(
        image_embeddings=image_embeddings,
        query_embeddings=query_embeddings,
        n_patches=n_patches,
        image_mask=image_mask,
    )

    # Get the similarity map for our input image
    similarity_maps = similarity_maps_batch[
        0
    ]  # (query_length, n_patches_x, n_patches_y)
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

    # Clean tokens for display (remove special characters that may cause encoding issues)
    display_tokens = [t.replace("Ġ", " ").replace("▁", " ") for t in filtered_tokens]
    print(f"\nQuery tokens: {display_tokens}")
    print(
        f"Similarity range: [{similarity_maps.min().item():.3f}, {similarity_maps.max().item():.3f}]"
    )

    # Generate all similarity maps
    print("\nGenerating similarity maps for all tokens...")
    plots = plot_all_similarity_maps(
        image=image,
        query_tokens=filtered_tokens,
        similarity_maps=similarity_maps,
        figsize=(8, 8),
        show_colorbar=False,
        add_title=True,
    )

    # Save the plots
    output_dir = Path("outputs/interpretability/colmodernvbert/" + uuid.uuid4().hex[:8])
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
        fig.savefig(savepath, bbox_inches="tight")
        print(f"  Saved: {savepath.name}")
        plt.close(fig)

    print(f"\n[SUCCESS] All similarity maps saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
