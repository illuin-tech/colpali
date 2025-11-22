#!/usr/bin/env python3
"""
Generate interpretability maps for ColModernVBert model using real documents.

This script demonstrates how to visualize which image regions the model
attends to for different query tokens, accounting for:
1. Idefics3 image splitting into sub-patches
2. Global patch exclusion for spatial correspondence
3. Correct token ordering across sub-patches

Usage:
    python examples/interpretability/colmodernvbert/generate_interpretability_maps.py
"""

from pathlib import Path

import torch
from datasets import load_dataset

from colpali_engine.interpretability.similarity_maps import plot_all_similarity_maps
from colpali_engine.models import ColModernVBert, ColModernVBertProcessor


def process_document(image, image_name, queries, processor, model, output_dir):
    """Process a document and generate interpretability maps for given queries."""
    print(f"\n{'='*60}")
    print(f"Processing: {image_name}")
    print(f"{'='*60}")

    # Save document
    image.save(output_dir / f"{image_name}.png")

    # Process image
    batch_images = processor.process_images([image])

    # Get patch dimensions (accounts for image splitting)
    # Note: patch_size parameter is optional for ColModernVBert (unused)
    n_patches_x, n_patches_y = processor.get_n_patches(
        (image.size[1], image.size[0])  # (height, width)
    )
    print(f"Patch grid: {n_patches_x} x {n_patches_y}")

    # Compute image embeddings
    with torch.no_grad():
        image_embeddings = model(**batch_images)

    # Get LOCAL image mask (excludes global patch)
    local_image_mask = processor.get_local_image_mask(batch_images)
    print(f"Local tokens: {int(local_image_mask.sum().item())}")

    print("\nGenerating similarity maps...")
    for query in queries:
        print(f"  Query: '{query}'", end=" ")
        batch_queries = processor.process_queries([query])

        with torch.no_grad():
            query_embeddings = model(**batch_queries)

        # Use processor's method for correct sub-patch ordering
        similarity_maps = processor.get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=(n_patches_x, n_patches_y),
            image_mask=local_image_mask,
        )

        sim_map = similarity_maps[0]

        # Filter out special tokens to avoid meaningless similarity noise
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

        if not filtered_tokens:
            print("(skipped - no tokens)")
            continue

        sim_map = sim_map[filtered_indices]
        print(f"-> Similarity: [{sim_map.min().item():.3f}, {sim_map.max().item():.3f}]")

        # Generate plots
        plots = plot_all_similarity_maps(image, filtered_tokens, sim_map)

        # Save
        # Sanitize filename by removing invalid characters
        query_safe = query.replace(" ", "_").replace("?", "").replace("/", "_").replace("\\", "_").replace(":", "")
        for idx, (fig, ax) in enumerate(plots):
            token = filtered_tokens[idx] if idx < len(filtered_tokens) else f"token_{idx}"
            token_safe = (token or f"token_{idx}").replace("<", "").replace(">", "").replace("Ġ", "").replace("?", "")
            fig.savefig(
                output_dir / f"{image_name}_{query_safe}_{idx}_{token_safe}.png",
                dpi=150,
                bbox_inches="tight",
            )
            fig.clf()


def main():
    print("=== ColModernVBert Interpretability Maps ===")
    print("Using real documents from datasets library\n")

    # Output directory (uses /outputs/ which is already in .gitignore)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    output_dir = project_root / "outputs" / "colmodernvbert_interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and processor
    print("Loading model and processor...")
    processor = ColModernVBertProcessor.from_pretrained("ModernVBERT/colmodernvbert")
    model = ColModernVBert.from_pretrained("ModernVBERT/colmodernvbert")
    model.eval()

    # Load real documents from DocVQA dataset
    print("Loading DocVQA dataset...")
    dataset = load_dataset("vidore/docvqa_test_subsampled", split="test", streaming=True)

    # Process first 10 documents for comprehensive testing
    num_samples = 10
    print(f"Processing first {num_samples} samples...\n")

    for idx, sample in enumerate(dataset):
        if idx >= num_samples:
            break

        image = sample["image"]
        query = sample["query"]

        print(f"\nSample {idx + 1}/{num_samples}")
        print(f"Query from dataset: '{query}'")

        # Process with the actual query from the dataset
        process_document(
            image,
            f"docvqa_sample_{idx + 1}",
            [query],
            processor,
            model,
            output_dir
        )

    print("\n" + "="*60)
    print("=== Done! ===")
    print(f"Visualizations saved to: {output_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()
