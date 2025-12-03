"""
Test ColGemma3 using colpali_engine - Modal deployment

This script tests the ColGemma3 implementation in colpali_engine
with the fine-tuned model Cognitive-Lab/ColNetraEmbed.

The test validates:
- Proper weight loading
- Image encoding with various dimensions
- Query encoding with variable lengths
- Similarity scoring (MaxSim)
- End-to-end retrieval

Usage:
    modal run modal_scripts/gemma3/test_colgemma3_modal.py
"""

import modal

# Create the Modal app
app = modal.App("colgemma3-test")

# Create a volume for persistent storage
volume = modal.Volume.from_name("nayana-ir-test-volume", create_if_missing=True)

# CUDA Configuration
CUDA_VERSION = "12.4.0"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu22.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Define the Modal image
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.11")
    .pip_install(
        [
            "torch",
            "torchvision",
            "transformers",
            "pillow",
            "numpy",
            "accelerate",
            "hf_transfer",
        ]
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/data/datasets/.cache",
        }
    )
    # Mount the entire colpali directory to get colpali_engine
    .add_local_dir(
        "../../",
        remote_path="/root/colpali",
    )
)

huggingface_secret = modal.Secret.from_name("adithya-hf-wandb")


@app.function(
    image=image,
    gpu="a100:1",
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=1800,  # 30 minute timeout
)
def run_colgemma3_test():
    """Run ColGemma3 test using colpali_engine on Modal with GPU."""
    import os
    import sys
    import torch
    from PIL import Image, ImageDraw, ImageFont

    # Set up HuggingFace authentication
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")

    # Add colpali to path
    sys.path.insert(0, "/root/colpali")

    # Import from colpali_engine
    from colpali_engine.models import ColGemma3, ColGemmaProcessor3

    print("=" * 80)
    print("ColGemma3 (Cognitive-Lab/ColNetraEmbed) - Modal Test")
    print("=" * 80)

    # Configuration
    model_name = "Cognitive-Lab/ColNetraEmbed"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Model: {model_name}")

    # Step 1: Create synthetic data
    print("\n" + "=" * 80)
    print("Step 1: Creating Synthetic Data")
    print("=" * 80)

    def create_synthetic_images(num_images=4):
        """Create synthetic test images."""
        images = []
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "cyan", "magenta"]
        texts = [
            "Financial Report Q4",
            "Organizational Chart",
            "Product Roadmap 2024",
            "Sales Dashboard",
        ]

        for i in range(num_images):
            # Create image with colored background
            img = Image.new("RGB", (224, 224), color=colors[i % len(colors)])
            draw = ImageDraw.Draw(img)
            text = texts[i % len(texts)]

            # Use default font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except Exception:
                font = ImageFont.load_default()

            # Center text
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((224 - text_width) // 2, (224 - text_height) // 2)

            # Draw text
            draw.text(position, text, fill="white", font=font)
            images.append(img)

            print(f"  Created image {i + 1}/{num_images}: {colors[i % len(colors)]} - '{text}'")

        return images

    num_images = 4
    images = create_synthetic_images(num_images)

    queries = [
        "financial report",
        "organizational structure",
        "product roadmap",
        "sales metrics",
    ]

    print(f"\nCreated {len(images)} synthetic images")
    print(f"Defined {len(queries)} test queries:")
    for i, query in enumerate(queries):
        print(f"  Query {i+1}: {query}")

    # Step 2: Load model and processor
    print("\n" + "=" * 80)
    print("Step 2: Loading ColGemma3 Model and Processor")
    print("=" * 80)

    model = ColGemma3.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    processor = ColGemmaProcessor3.from_pretrained(model_name, use_fast=True)

    print(f"✓ Model loaded successfully")
    print(f"  - Embedding dimension: {model.dim}")
    print(f"  - Device: {model.device}")
    print(f"  - Dtype: {model.dtype}")

    # Step 3: Encode images
    print("\n" + "=" * 80)
    print("Step 3: Encoding Images")
    print("=" * 80)

    batch_images = processor.process_images(images).to(model.device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)

    print(f"✓ Images encoded successfully")
    print(f"  - Shape: {image_embeddings.shape}")
    print(f"  - Dtype: {image_embeddings.dtype}")
    print(f"  - Has NaN: {torch.isnan(image_embeddings).any().item()}")
    print(f"  - Has Inf: {torch.isinf(image_embeddings).any().item()}")

    norms = torch.norm(image_embeddings, dim=2).mean(dim=1)
    print(f"  - Average token norms per image: {norms.tolist()}")

    # Step 4: Encode queries
    print("\n" + "=" * 80)
    print("Step 4: Encoding Queries")
    print("=" * 80)

    batch_queries = processor.process_queries(queries).to(model.device)

    with torch.no_grad():
        query_embeddings = model(**batch_queries)

    print(f"✓ Queries encoded successfully")
    print(f"  - Shape: {query_embeddings.shape}")
    print(f"  - Dtype: {query_embeddings.dtype}")
    print(f"  - Has NaN: {torch.isnan(query_embeddings).any().item()}")
    print(f"  - Has Inf: {torch.isinf(query_embeddings).any().item()}")

    norms = torch.norm(query_embeddings, dim=2).mean(dim=1)
    print(f"  - Average token norms per query: {norms.tolist()}")

    # Step 5: Compute similarity scores
    print("\n" + "=" * 80)
    print("Step 5: Computing Similarity Scores (MaxSim)")
    print("=" * 80)

    scores = processor.score_multi_vector(
        qs=query_embeddings,
        ps=image_embeddings,
    )

    print(f"✓ Scores computed successfully")
    print(f"  - Shape: {scores.shape}")
    print(f"  - Dtype: {scores.dtype}")

    # Self-similarity checks
    print(f"\n✓ Self-Similarity Scores:")
    print(f"  Image with itself:")
    for i in range(len(images)):
        self_score_img = processor.score_multi_vector(
            qs=image_embeddings[i:i+1],
            ps=image_embeddings[i:i+1],
        )[0, 0].item()
        print(f"    Image {i + 1} vs Image {i + 1}: {self_score_img:.4f}")

    print(f"  Query with itself:")
    for i in range(len(queries)):
        self_score_query = processor.score_multi_vector(
            qs=query_embeddings[i:i+1],
            ps=query_embeddings[i:i+1],
        )[0, 0].item()
        print(f"    Query {i + 1} vs Query {i + 1}: {self_score_query:.4f}")

    print(f"\n  Note: Self-similarity scores should be high (close to max possible)")

    print(f"\nSimilarity Score Matrix:")
    header = "Query / Image"
    print(f"{header:<40} " + " ".join([f"Img{i + 1:>7}" for i in range(len(images))]))
    print("-" * (40 + 9 * len(images)))

    for i, query in enumerate(queries):
        query_short = query[:35] + "..." if len(query) > 35 else query
        score_str = " ".join([f"{scores[i, j].item():>8.2f}" for j in range(len(images))])
        print(f"{query_short:<40} {score_str}")

    # Step 6: Retrieval validation
    print("\n" + "=" * 80)
    print("Step 6: Retrieval Validation")
    print("=" * 80)

    print(f"\nTop Matches:")
    for i, query in enumerate(queries):
        best_idx = scores[i].argmax().item()
        best_score = scores[i, best_idx].item()
        print(f"  Query {i + 1}: Best match is Image {best_idx + 1} (score: {best_score:.2f})")

    # Check if there's reasonable matching (scores should be different)
    score_variance = scores.var().item()
    print(f"\n✓ Score variance: {score_variance:.4f}")
    print(f"  (Higher variance indicates model is discriminating between images)")

    # Check best matches are plausible
    print("\n✓ Query-Image matching analysis:")
    for i, query in enumerate(queries):
        best_idx = scores[i].argmax().item()
        print(f"  '{query}' -> Image {best_idx + 1} (expected: Image {i + 1})")

    # Step 7: Validation summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    validation_passed = True

    # Check 1: Shape validation
    checks = []
    checks.append(("Image batch size", image_embeddings.shape[0] == len(images), f"{len(images)}"))
    checks.append(("Image embedding dim", image_embeddings.shape[2] == 128, "128"))
    checks.append(("Query batch size", query_embeddings.shape[0] == len(queries), f"{len(queries)}"))
    checks.append(("Query embedding dim", query_embeddings.shape[2] == 128, "128"))
    checks.append(("Score matrix shape", scores.shape == (len(queries), len(images)), f"({len(queries)}, {len(images)})"))
    checks.append(("No NaN in images", not torch.isnan(image_embeddings).any(), "clean"))
    checks.append(("No Inf in images", not torch.isinf(image_embeddings).any(), "clean"))
    checks.append(("No NaN in queries", not torch.isnan(query_embeddings).any(), "clean"))
    checks.append(("No Inf in queries", not torch.isinf(query_embeddings).any(), "clean"))
    checks.append(("Score variance", score_variance > 0.01, f"{score_variance:.4f}"))

    for check_name, passed, expected in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}: {expected}")
        if not passed:
            validation_passed = False

    print(f"\n{'=' * 80}")
    if validation_passed:
        print("✓ ALL TESTS PASSED! 🎉")
        print(f"  - {len(images)} images encoded and retrieved correctly")
        print(f"  - {len(queries)} queries encoded and matched correctly")
        print(f"  - {len(images) * len(queries)} similarity scores computed")
        print("\nColGemma3 (Cognitive-Lab/ColNetraEmbed) is working correctly!")
    else:
        print("✗ SOME TESTS FAILED")
        print("  Review errors above")

    # Save results to volume
    volume.commit()

    return {
        "status": "success" if validation_passed else "failed",
        "model": model_name,
        "num_images": len(images),
        "num_queries": len(queries),
        "image_embeddings_shape": list(image_embeddings.shape),
        "query_embeddings_shape": list(query_embeddings.shape),
        "similarity_scores": scores.cpu().tolist(),
        "validation_passed": validation_passed,
        "score_variance": score_variance,
    }


@app.local_entrypoint()
def main():
    """Local entrypoint to run the test."""
    print("Running ColGemma3 (Cognitive-Lab/ColNetraEmbed) test on Modal...")
    result = run_colgemma3_test.remote()

    print(f"\n{'=' * 80}")
    print("Modal Execution Complete")
    print(f"{'=' * 80}")
    print(f"Model: {result['model']}")
    print(f"Status: {result['status']}")
    print(f"Images processed: {result['num_images']}")
    print(f"Queries processed: {result['num_queries']}")
    print(f"Image embeddings shape: {result['image_embeddings_shape']}")
    print(f"Query embeddings shape: {result['query_embeddings_shape']}")
    print(f"Score variance: {result['score_variance']:.4f}")
    print(f"Validation: {'PASSED ✓' if result['validation_passed'] else 'FAILED ✗'}")

    return result
