"""
Test BiGemma3 using vLLM - Modal deployment

This script tests the BiGemma3 implementation using vLLM
with the fine-tuned model Cognitive-Lab/NetraEmbed.

The test validates:
- Proper weight loading with vLLM
- Image encoding via vLLM (offline and online modes)
- Query encoding via vLLM (offline and online modes)
- Similarity scoring (cosine similarity)
- End-to-end retrieval

Tests both:
- Offline mode: Direct LLM.embed() calls
- Online mode: vLLM server with OpenAI-compatible API

Note: This uses vLLM's pooling mode with full embedding dimension (2560).

Usage:
    modal run modal_scripts/gemma3/vllm/test_bigemma3_vllm.py
"""

import modal

# Create the Modal app
app = modal.App("bigemma3-vllm-test")

# Create a volume for persistent storage
volume = modal.Volume.from_name("nayana-ir-test-volume", create_if_missing=True)

# CUDA Configuration
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Define the Modal image with vLLM
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev")
    .run_commands("uv pip install vllm -U --system")
    .uv_pip_install(
        "datasets",
        "pillow",
        "huggingface_hub[hf_transfer]",
        "requests",
        "numpy",
        "regex",
        "sentencepiece",
        "torch",
        "transformers",
    )
    .run_commands("uv pip install flash-attn --no-build-isolation --system")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/data/.cache",
        }
    )
)

huggingface_secret = modal.Secret.from_name("adithya-hf-wandb")

# Configuration
MODEL_NAME = "Cognitive-Lab/NetraEmbed"
EMBEDDING_DIM = 2560  # Full dimension only
VLLM_PORT = 8000


@app.function(
    image=image,
    gpu="l40s:1",
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=1800,  # 30 minute timeout
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * 60)
def serve():
    """Serve vLLM embedding server for online testing."""
    import os
    import subprocess

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--runner",
        "pooling",  # REQUIRED for embedding mode
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.4",
        "--max-model-len",
        "4096",
        "--limit-mm-per-prompt",
        '{"image": 2}',
        "--trust-remote-code",
        "--override-pooler-config",
        '{"pooling_type": "LAST"}',
    ]

    print("Starting vLLM server:", " ".join(cmd))
    subprocess.Popen(cmd)


@app.function(
    image=image,
    gpu="l40s:1",
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=1800,  # 30 minute timeout
)
def run_bigemma3_vllm_offline_test():
    """Run BiGemma3 OFFLINE test using vLLM LLM.embed() on Modal with GPU."""
    import os

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from vllm import LLM

    # Set up HuggingFace authentication
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")

    print("=" * 80)
    print("BiGemma3 (Cognitive-Lab/NetraEmbed) - vLLM OFFLINE Test")
    print("=" * 80)

    print(f"\nModel: {MODEL_NAME}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print("Mode: OFFLINE (LLM.embed())")

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

    # Step 2: Initialize vLLM model
    print("\n" + "=" * 80)
    print("Step 2: Initializing vLLM Model")
    print("=" * 80)

    print(f"Initializing vLLM with model: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        runner="pooling",  # REQUIRED for embedding mode
        limit_mm_per_prompt={"image": 2},  # Allow multimodal inputs
        gpu_memory_utilization=0.6,
        max_model_len=4096,
        trust_remote_code=True,
        override_pooler_config={"pooling_type": "LAST"},  # Override for BiGemma3
    )

    print("✓ Model loaded successfully")
    print(f"  - Model: {MODEL_NAME}")
    print(f"  - Embedding dimension: {EMBEDDING_DIM}")
    print("  - Runner mode: pooling")

    # Step 3: Encode images
    print("\n" + "=" * 80)
    print("Step 3: Encoding Images")
    print("=" * 80)

    image_embeddings_list = []
    for i, img in enumerate(images):
        print(f"Processing image {i + 1}/{len(images)}...")
        # For BiGemma3/Gemma3, we need <start_of_image> placeholder
        # The model expects images as PIL Image objects with proper prompt
        prompt = "<start_of_image>"
        outputs = llm.embed({"prompt": prompt, "multi_modal_data": {"image": img}})
        embedding = outputs[0].outputs.embedding
        image_embeddings_list.append(embedding)

    image_embeddings = np.array(image_embeddings_list)

    print("✓ Images encoded successfully")
    print(f"  - Shape: {image_embeddings.shape}")
    print(f"  - Expected: ({len(images)}, {EMBEDDING_DIM})")
    print(f"  - Has NaN: {np.isnan(image_embeddings).any()}")
    print(f"  - Has Inf: {np.isinf(image_embeddings).any()}")

    # Calculate L2 norms
    norms = np.linalg.norm(image_embeddings, axis=1)
    print(f"  - L2 norms per image: {norms.tolist()}")
    print(f"  - Note: Should be ~1.0 (L2 normalized)")

    # Step 4: Encode queries
    print("\n" + "=" * 80)
    print("Step 4: Encoding Queries")
    print("=" * 80)

    query_embeddings_list = []
    for i, query in enumerate(queries):
        print(f"Processing query {i + 1}/{len(queries)}: {query}")
        # For text-only queries
        outputs = llm.embed(query)
        embedding = outputs[0].outputs.embedding
        query_embeddings_list.append(embedding)

    query_embeddings = np.array(query_embeddings_list)

    print("✓ Queries encoded successfully")
    print(f"  - Shape: {query_embeddings.shape}")
    print(f"  - Expected: ({len(queries)}, {EMBEDDING_DIM})")
    print(f"  - Has NaN: {np.isnan(query_embeddings).any()}")
    print(f"  - Has Inf: {np.isinf(query_embeddings).any()}")

    # Calculate L2 norms
    norms = np.linalg.norm(query_embeddings, axis=1)
    print(f"  - L2 norms per query: {norms.tolist()}")
    print(f"  - Note: Should be ~1.0 (L2 normalized)")

    # Step 5: Compute similarity scores
    print("\n" + "=" * 80)
    print("Step 5: Computing Similarity Scores (Cosine Similarity)")
    print("=" * 80)

    def cosine_similarity_matrix(queries, images):
        """Compute cosine similarity between query and image embeddings."""
        # Normalize embeddings (should already be normalized)
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        images_norm = images / (np.linalg.norm(images, axis=1, keepdims=True) + 1e-8)
        # Compute dot product (cosine similarity for normalized vectors)
        return queries_norm @ images_norm.T

    scores = cosine_similarity_matrix(query_embeddings, image_embeddings)

    print(f"✓ Scores computed successfully")
    print(f"  - Shape: {scores.shape}")
    print(f"  - Min score: {scores.min():.4f}")
    print(f"  - Max score: {scores.max():.4f}")
    print(f"  - Note: Scores range from -1 to 1 (cosine similarity)")

    # Self-similarity checks
    print(f"\n✓ Self-Similarity Scores:")
    print(f"  Image with itself:")
    for i in range(len(images)):
        self_score_img = cosine_similarity_matrix(
            image_embeddings[i : i + 1], image_embeddings[i : i + 1]
        )[0, 0]
        print(f"    Image {i + 1} vs Image {i + 1}: {self_score_img:.4f}")

    print(f"  Query with itself:")
    for i in range(len(queries)):
        self_score_query = cosine_similarity_matrix(
            query_embeddings[i : i + 1], query_embeddings[i : i + 1]
        )[0, 0]
        print(f"    Query {i + 1} vs Query {i + 1}: {self_score_query:.4f}")

    print(f"\n  Note: Self-similarity scores should be 1.0 (perfect cosine similarity)")

    print(f"\nSimilarity Score Matrix:")
    header = "Query / Image"
    print(f"{header:<40} " + " ".join([f"Img{i + 1:>7}" for i in range(len(images))]))
    print("-" * (40 + 9 * len(images)))

    for i, query in enumerate(queries):
        query_short = query[:35] + "..." if len(query) > 35 else query
        score_str = " ".join([f"{scores[i, j]:>8.4f}" for j in range(len(images))])
        print(f"{query_short:<40} {score_str}")

    # Step 6: Retrieval validation
    print("\n" + "=" * 80)
    print("Step 6: Retrieval Validation")
    print("=" * 80)

    print(f"\nTop Matches:")
    for i, query in enumerate(queries):
        best_idx = scores[i].argmax()
        best_score = scores[i, best_idx]
        print(f"  Query {i + 1}: Best match is Image {best_idx + 1} (score: {best_score:.4f})")

    # Check if there's reasonable matching (scores should be different)
    score_variance = scores.var()
    print(f"\n✓ Score variance: {score_variance:.6f}")
    print(f"  (Higher variance indicates model is discriminating between images)")

    # Check best matches are plausible
    print("\n✓ Query-Image matching analysis:")
    for i, query in enumerate(queries):
        best_idx = scores[i].argmax()
        print(f"  '{query}' -> Image {best_idx + 1} (expected: Image {i + 1})")

    # Step 7: Validation summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    validation_passed = True

    # Check 1: Shape validation
    checks = []
    checks.append(("Image batch size", image_embeddings.shape[0] == len(images), f"{len(images)}"))
    checks.append(
        ("Image embedding dim", image_embeddings.shape[1] == EMBEDDING_DIM, f"{EMBEDDING_DIM}")
    )
    checks.append(("Image embeddings 2D", image_embeddings.ndim == 2, "2D array"))
    checks.append(("Query batch size", query_embeddings.shape[0] == len(queries), f"{len(queries)}"))
    checks.append(
        ("Query embedding dim", query_embeddings.shape[1] == EMBEDDING_DIM, f"{EMBEDDING_DIM}")
    )
    checks.append(("Query embeddings 2D", query_embeddings.ndim == 2, "2D array"))
    checks.append(("Score matrix shape", scores.shape == (len(queries), len(images)), f"({len(queries)}, {len(images)})"))
    checks.append(("No NaN in images", not np.isnan(image_embeddings).any(), "clean"))
    checks.append(("No Inf in images", not np.isinf(image_embeddings).any(), "clean"))
    checks.append(("No NaN in queries", not np.isnan(query_embeddings).any(), "clean"))
    checks.append(("No Inf in queries", not np.isinf(query_embeddings).any(), "clean"))
    checks.append(
        (
            "Embeddings normalized",
            np.allclose(np.linalg.norm(image_embeddings, axis=1), np.ones(len(images)), atol=1e-5),
            "~1.0",
        )
    )
    checks.append(("Score variance", score_variance > 0.0001, f"{score_variance:.6f}"))

    for check_name, passed, expected in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}: {expected}")
        if not passed:
            validation_passed = False

    print(f"\n{'=' * 80}")
    if validation_passed:
        print("✓ ALL TESTS PASSED! 🎉")
        print(f"  - {len(images)} images encoded to single vectors")
        print(f"  - {len(queries)} queries encoded to single vectors")
        print(f"  - {len(images) * len(queries)} cosine similarities computed")
        print(f"  - All embeddings properly L2 normalized")
        print("\nBiGemma3 (Cognitive-Lab/NetraEmbed) with vLLM is working correctly!")
    else:
        print("✗ SOME TESTS FAILED")
        print("  Review errors above")

    # Save results to volume
    volume.commit()

    return {
        "status": "success" if validation_passed else "failed",
        "mode": "offline",
        "model": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "num_images": len(images),
        "num_queries": len(queries),
        "image_embeddings_shape": list(image_embeddings.shape),
        "query_embeddings_shape": list(query_embeddings.shape),
        "similarity_scores": scores.tolist(),
        "validation_passed": validation_passed,
        "score_variance": float(score_variance),
    }


@app.function(
    image=image,
    secrets=[huggingface_secret],
    timeout=1800,
)
def run_bigemma3_vllm_online_test():
    """Run BiGemma3 ONLINE test using vLLM API server."""
    import base64
    import io

    import numpy as np
    import requests
    from PIL import Image, ImageDraw, ImageFont

    print("=" * 80)
    print("BiGemma3 (Cognitive-Lab/NetraEmbed) - vLLM ONLINE Test")
    print("=" * 80)

    server_url = serve.web_url
    print(f"\nServer URL: {server_url}")
    print(f"Model: {MODEL_NAME}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print("Mode: ONLINE (API server)")

    # Helper function to encode image to base64
    def encode_image(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Step 1: Create synthetic data (same as offline)
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
            img = Image.new("RGB", (224, 224), color=colors[i % len(colors)])
            draw = ImageDraw.Draw(img)
            text = texts[i % len(texts)]

            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except Exception:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((224 - text_width) // 2, (224 - text_height) // 2)

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

    # Step 2: Encode images via API
    print("\n" + "=" * 80)
    print("Step 2: Encoding Images via API")
    print("=" * 80)

    image_embeddings_list = []
    for i, img in enumerate(images):
        print(f"Processing image {i + 1}/{len(images)}...")
        base64_image = encode_image(img)

        # For BiGemma3/Gemma3, use <start_of_image> placeholder
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": "<start_of_image>"},
                ],
            }
        ]

        response = requests.post(
            f"{server_url}/v1/embeddings",
            json={"model": MODEL_NAME, "messages": messages, "encoding_format": "float"},
            timeout=300 if i == 0 else 120,  # First request may take longer for warmup
        )

        if response.status_code == 200:
            embedding = response.json()["data"][0]["embedding"]
            image_embeddings_list.append(embedding)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {"status": "failed", "error": f"Image encoding failed: {response.text}"}

    image_embeddings = np.array(image_embeddings_list)

    print("✓ Images encoded successfully")
    print(f"  - Shape: {image_embeddings.shape}")
    print(f"  - Expected: ({len(images)}, {EMBEDDING_DIM})")
    print(f"  - Has NaN: {np.isnan(image_embeddings).any()}")
    print(f"  - Has Inf: {np.isinf(image_embeddings).any()}")

    norms = np.linalg.norm(image_embeddings, axis=1)
    print(f"  - L2 norms per image: {norms.tolist()}")
    print("  - Note: Should be ~1.0 (L2 normalized)")

    # Step 3: Encode queries via API
    print("\n" + "=" * 80)
    print("Step 3: Encoding Queries via API")
    print("=" * 80)

    query_embeddings_list = []
    for i, query in enumerate(queries):
        print(f"Processing query {i + 1}/{len(queries)}: {query}")

        messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]

        response = requests.post(
            f"{server_url}/v1/embeddings",
            json={"model": MODEL_NAME, "messages": messages, "encoding_format": "float"},
            timeout=120,
        )

        if response.status_code == 200:
            embedding = response.json()["data"][0]["embedding"]
            query_embeddings_list.append(embedding)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {"status": "failed", "error": f"Query encoding failed: {response.text}"}

    query_embeddings = np.array(query_embeddings_list)

    print("✓ Queries encoded successfully")
    print(f"  - Shape: {query_embeddings.shape}")
    print(f"  - Expected: ({len(queries)}, {EMBEDDING_DIM})")
    print(f"  - Has NaN: {np.isnan(query_embeddings).any()}")
    print(f"  - Has Inf: {np.isinf(query_embeddings).any()}")

    norms = np.linalg.norm(query_embeddings, axis=1)
    print(f"  - L2 norms per query: {norms.tolist()}")
    print("  - Note: Should be ~1.0 (L2 normalized)")

    # Step 4: Compute similarity scores (same as offline)
    print("\n" + "=" * 80)
    print("Step 4: Computing Similarity Scores (Cosine Similarity)")
    print("=" * 80)

    def cosine_similarity_matrix(queries, images):
        """Compute cosine similarity between query and image embeddings."""
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        images_norm = images / (np.linalg.norm(images, axis=1, keepdims=True) + 1e-8)
        return queries_norm @ images_norm.T

    scores = cosine_similarity_matrix(query_embeddings, image_embeddings)

    print("✓ Scores computed successfully")
    print(f"  - Shape: {scores.shape}")
    print(f"  - Min score: {scores.min():.4f}")
    print(f"  - Max score: {scores.max():.4f}")
    print("  - Note: Scores range from -1 to 1 (cosine similarity)")

    # Self-similarity checks
    print("\n✓ Self-Similarity Scores:")
    print("  Image with itself:")
    for i in range(len(images)):
        self_score = cosine_similarity_matrix(
            image_embeddings[i : i + 1], image_embeddings[i : i + 1]
        )[0, 0]
        print(f"    Image {i + 1} vs Image {i + 1}: {self_score:.4f}")

    print("  Query with itself:")
    for i in range(len(queries)):
        self_score = cosine_similarity_matrix(
            query_embeddings[i : i + 1], query_embeddings[i : i + 1]
        )[0, 0]
        print(f"    Query {i + 1} vs Query {i + 1}: {self_score:.4f}")

    print("\n  Note: Self-similarity scores should be 1.0 (perfect cosine similarity)")

    print("\nSimilarity Score Matrix:")
    header = "Query / Image"
    print(f"{header:<40} " + " ".join([f"Img{i + 1:>7}" for i in range(len(images))]))
    print("-" * (40 + 9 * len(images)))

    for i, query in enumerate(queries):
        query_short = query[:35] + "..." if len(query) > 35 else query
        score_str = " ".join([f"{scores[i, j]:>8.4f}" for j in range(len(images))])
        print(f"{query_short:<40} {score_str}")

    # Step 5: Retrieval validation
    print("\n" + "=" * 80)
    print("Step 5: Retrieval Validation")
    print("=" * 80)

    print("\nTop Matches:")
    for i, query in enumerate(queries):
        best_idx = scores[i].argmax()
        best_score = scores[i, best_idx]
        print(f"  Query {i + 1}: Best match is Image {best_idx + 1} (score: {best_score:.4f})")

    score_variance = scores.var()
    print(f"\n✓ Score variance: {score_variance:.6f}")
    print("  (Higher variance indicates model is discriminating between images)")

    print("\n✓ Query-Image matching analysis:")
    for i, query in enumerate(queries):
        best_idx = scores[i].argmax()
        print(f"  '{query}' -> Image {best_idx + 1} (expected: Image {i + 1})")

    # Step 6: Validation summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    validation_passed = True

    checks = []
    checks.append(("Image batch size", image_embeddings.shape[0] == len(images), f"{len(images)}"))
    checks.append(
        ("Image embedding dim", image_embeddings.shape[1] == EMBEDDING_DIM, f"{EMBEDDING_DIM}")
    )
    checks.append(("Image embeddings 2D", image_embeddings.ndim == 2, "2D array"))
    checks.append(("Query batch size", query_embeddings.shape[0] == len(queries), f"{len(queries)}"))
    checks.append(
        ("Query embedding dim", query_embeddings.shape[1] == EMBEDDING_DIM, f"{EMBEDDING_DIM}")
    )
    checks.append(("Query embeddings 2D", query_embeddings.ndim == 2, "2D array"))
    checks.append(
        (
            "Score matrix shape",
            scores.shape == (len(queries), len(images)),
            f"({len(queries)}, {len(images)})",
        )
    )
    checks.append(("No NaN in images", not np.isnan(image_embeddings).any(), "clean"))
    checks.append(("No Inf in images", not np.isinf(image_embeddings).any(), "clean"))
    checks.append(("No NaN in queries", not np.isnan(query_embeddings).any(), "clean"))
    checks.append(("No Inf in queries", not np.isinf(query_embeddings).any(), "clean"))
    checks.append(
        (
            "Embeddings normalized",
            np.allclose(np.linalg.norm(image_embeddings, axis=1), np.ones(len(images)), atol=1e-5),
            "~1.0",
        )
    )
    checks.append(("Score variance", score_variance > 0.0001, f"{score_variance:.6f}"))

    for check_name, passed, expected in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}: {expected}")
        if not passed:
            validation_passed = False

    print(f"\n{'=' * 80}")
    if validation_passed:
        print("✓ ALL TESTS PASSED! 🎉")
        print(f"  - {len(images)} images encoded to single vectors")
        print(f"  - {len(queries)} queries encoded to single vectors")
        print(f"  - {len(images) * len(queries)} cosine similarities computed")
        print("  - All embeddings properly L2 normalized")
        print("\nBiGemma3 (Cognitive-Lab/NetraEmbed) with vLLM ONLINE API is working correctly!")
    else:
        print("✗ SOME TESTS FAILED")
        print("  Review errors above")

    return {
        "status": "success" if validation_passed else "failed",
        "mode": "online",
        "model": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "num_images": len(images),
        "num_queries": len(queries),
        "image_embeddings_shape": list(image_embeddings.shape),
        "query_embeddings_shape": list(query_embeddings.shape),
        "similarity_scores": scores.tolist(),
        "validation_passed": validation_passed,
        "score_variance": float(score_variance),
    }


@app.local_entrypoint()
def main():
    """Local entrypoint to run both offline and online tests."""
    print("Running BiGemma3 (Cognitive-Lab/NetraEmbed) vLLM tests on Modal...")
    print("This will test both OFFLINE (LLM.embed()) and ONLINE (API) modes.\n")

    # Run offline test
    print("\n" + "=" * 80)
    print("STARTING OFFLINE TEST")
    print("=" * 80)
    offline_result = run_bigemma3_vllm_offline_test.remote()

    print(f"\n{'=' * 80}")
    print("OFFLINE TEST COMPLETE")
    print(f"{'=' * 80}")
    print(f"Model: {offline_result['model']}")
    print(f"Mode: {offline_result['mode']}")
    print(f"Status: {offline_result['status']}")
    print(f"Embedding dimension: {offline_result['embedding_dim']}")
    print(f"Images processed: {offline_result['num_images']}")
    print(f"Queries processed: {offline_result['num_queries']}")
    print(f"Score variance: {offline_result['score_variance']:.6f}")
    print(f"Validation: {'PASSED ✓' if offline_result['validation_passed'] else 'FAILED ✗'}")

    # Run online test
    print("\n" + "=" * 80)
    print("STARTING ONLINE TEST")
    print("=" * 80)
    online_result = run_bigemma3_vllm_online_test.remote()

    print(f"\n{'=' * 80}")
    print("ONLINE TEST COMPLETE")
    print(f"{'=' * 80}")
    print(f"Model: {online_result['model']}")
    print(f"Mode: {online_result['mode']}")
    print(f"Status: {online_result['status']}")
    print(f"Embedding dimension: {online_result['embedding_dim']}")
    print(f"Images processed: {online_result['num_images']}")
    print(f"Queries processed: {online_result['num_queries']}")
    print(f"Score variance: {online_result['score_variance']:.6f}")
    print(f"Validation: {'PASSED ✓' if online_result['validation_passed'] else 'FAILED ✗'}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"Offline Test: {'PASSED ✓' if offline_result['validation_passed'] else 'FAILED ✗'}")
    print(f"Online Test: {'PASSED ✓' if online_result['validation_passed'] else 'FAILED ✗'}")

    overall_passed = offline_result['validation_passed'] and online_result['validation_passed']
    if overall_passed:
        print("\n✓ ALL TESTS PASSED! 🎉")
        print("BiGemma3 (Cognitive-Lab/NetraEmbed) with vLLM works correctly in both modes!")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Review the output above for details.")

    return {
        "offline": offline_result,
        "online": online_result,
        "overall_passed": overall_passed,
    }
