#!/usr/bin/env python3
"""
Generate interpretability maps for ColModernVBert model.

This script demonstrates how to visualize which image regions the model
attends to for different query tokens, accounting for:
1. Idefics3 image splitting into sub-patches
2. Global patch exclusion for spatial correspondence
3. Correct token ordering across sub-patches

Usage:
    python tests/models/modernvbert/generate_interpretability_maps.py
"""

import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from colpali_engine.interpretability.similarity_maps import plot_all_similarity_maps
from colpali_engine.models import ColModernVBert, ColModernVBertProcessor


def create_sample_document():
    """Create a synthetic document for demonstration."""
    width, height = 800, 1000
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype("arial.ttf", 32)
        font_medium = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    # Header
    draw.text((50, 30), "INVOICE", fill="#1E3A5F", font=font_large)
    draw.text((500, 30), "Invoice #: INV-2024-0847", fill="black", font=font_small)
    draw.text((500, 50), "Date: November 15, 2024", fill="black", font=font_small)
    draw.line([(50, 100), (750, 100)], fill="#1E3A5F", width=3)

    # Billing info
    draw.text((50, 120), "BILL TO:", fill="#1E3A5F", font=font_medium)
    draw.text((50, 145), "ABC Industries Ltd.", fill="black", font=font_medium)
    draw.text((50, 170), "456 Industrial Park", fill="black", font=font_small)

    # Items table
    table_y = 250
    draw.rectangle([50, table_y, 750, table_y + 30], fill="#E8E8E8")
    draw.text((60, table_y + 8), "Description", fill="black", font=font_medium)
    draw.text((450, table_y + 8), "Amount", fill="black", font=font_medium)

    items = [
        ("Laptop Computer", "$6,000.00"),
        ('Monitor 27" 4K', "$2,250.00"),
        ("Keyboard", "$750.00"),
    ]

    for i, (desc, amount) in enumerate(items):
        row_y = table_y + 30 + (i * 35)
        draw.text((60, row_y + 10), desc, fill="black", font=font_small)
        draw.text((450, row_y + 10), amount, fill="black", font=font_small)

    # Totals
    totals_y = table_y + 30 + len(items) * 35 + 30
    draw.text((450, totals_y), "TOTAL:", fill="#1E3A5F", font=font_medium)
    draw.text((550, totals_y), "$11,935.27", fill="#1E3A5F", font=font_medium)

    # Signature
    draw.text((50, 600), "Signature:", fill="black", font=font_small)
    signature_points = [(70, 650), (90, 640), (130, 660), (170, 645), (210, 655)]
    for i in range(len(signature_points) - 1):
        draw.line([signature_points[i], signature_points[i + 1]], fill="blue", width=2)

    return img


def main():
    print("=== ColModernVBert Interpretability Maps ===\n")

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

    # Create document
    print("Creating sample document...")
    image = create_sample_document()
    image.save(output_dir / "document.png")

    # Process image
    batch_images = processor.process_images([image])

    # Get patch dimensions (accounts for image splitting)
    n_patches_x, n_patches_y = processor.get_n_patches(
        (image.size[1], image.size[0]),  # (height, width)
        patch_size=14,  # API compatibility, not used
    )
    print(f"Patch grid: {n_patches_x} x {n_patches_y}")

    # Compute image embeddings
    with torch.no_grad():
        image_embeddings = model(**batch_images)

    # Get LOCAL image mask (excludes global patch)
    local_image_mask = processor.get_local_image_mask(batch_images)
    print(f"Local tokens: {int(local_image_mask.sum().item())}")

    # Test queries
    queries = ["total amount", "invoice number", "signature", "laptop"]

    print("\n=== Generating Similarity Maps ===")
    for query in queries:
        print(f"\nQuery: '{query}'")
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
            print("  Skipping query – no non-special tokens found.")
            continue

        sim_map = sim_map[filtered_indices]
        print(f"  Similarity: [{sim_map.min().item():.3f}, {sim_map.max().item():.3f}]")

        # Generate plots
        plots = plot_all_similarity_maps(image, filtered_tokens, sim_map)

        # Save
        query_safe = query.replace(" ", "_")
        for idx, (fig, ax) in enumerate(plots):
            token = filtered_tokens[idx] if idx < len(filtered_tokens) else f"token_{idx}"
            token_safe = (token or f"token_{idx}").replace("<", "").replace(">", "").replace("Ġ", "")
            fig.savefig(
                output_dir / f"{query_safe}_{idx}_{token_safe}.png",
                dpi=150,
                bbox_inches="tight",
            )
            fig.clf()

    print("\n=== Done! ===")
    print(f"Visualizations saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
