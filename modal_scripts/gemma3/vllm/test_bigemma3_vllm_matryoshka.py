"""
Test BiGemma3 with Matryoshka Embeddings using vLLM - Modal deployment

This script tests the BiGemma3 implementation using vLLM with Matryoshka embedding support.
It validates that the model works correctly with different embedding dimensions.

The test validates:
- Matryoshka embedding truncation (768, 1536, 2560 dimensions)
- Image encoding via vLLM (offline and online modes)
- Query encoding via vLLM (offline and online modes)
- Similarity scoring across different dimensions
- End-to-end retrieval performance at each dimension

Tests both:
- Offline mode: Direct LLM.embed() with PoolingParams(dimensions=X)
- Online mode: vLLM server API with dimensions parameter

Model: Cognitive-Lab/NetraEmbed (BiGemma3-based)

Usage:
    modal run modal_scripts/gemma3/vllm/test_bigemma3_vllm_matryoshka.py
"""

import modal

# Create the Modal app
app = modal.App("bigemma3-vllm-matryoshka-test")

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
MATRYOSHKA_DIMS = [768, 1536, 2560]  # Test all Matryoshka dimensions
VLLM_PORT = 8000


@app.function(
    image=image,
    gpu="l40s:1",
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=1800,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * 60)
def serve():
    """Serve vLLM embedding server for online Matryoshka testing."""
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
        # Override to enable Matryoshka support
        "--hf-overrides",
        f'{{"matryoshka_dimensions": {MATRYOSHKA_DIMS}}}',
    ]

    print("Starting vLLM server for Matryoshka embeddings:", " ".join(cmd))
    subprocess.Popen(cmd)


def create_synthetic_images(num_images=4):
    """Create synthetic test images - shared utility function."""
    from PIL import Image, ImageDraw, ImageFont

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


@app.function(
    image=image,
    gpu="l40s:1",
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=1800,
)
def run_matryoshka_offline_test():
    """Run BiGemma3 Matryoshka OFFLINE test using vLLM LLM.embed() with PoolingParams."""
    import os

    import numpy as np
    from vllm import LLM, PoolingParams

    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")

    print("=" * 80)
    print("BiGemma3 Matryoshka - vLLM OFFLINE Test")
    print("=" * 80)

    print(f"\nModel: {MODEL_NAME}")
    print(f"Matryoshka dimensions: {MATRYOSHKA_DIMS}")
    print("Mode: OFFLINE (LLM.embed() with PoolingParams)")

    # Step 1: Create synthetic data
    print("\n" + "=" * 80)
    print("Step 1: Creating Synthetic Data")
    print("=" * 80)

    images = create_synthetic_images(num_images=4)

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
        runner="pooling",
        limit_mm_per_prompt={"image": 2},
        gpu_memory_utilization=0.6,
        max_model_len=4096,
        trust_remote_code=True,
        override_pooler_config={"pooling_type": "LAST"},
        # Override to enable Matryoshka support
        hf_overrides={"matryoshka_dimensions": MATRYOSHKA_DIMS},
    )

    print("✓ Model loaded successfully")
    print(f"  - Model: {MODEL_NAME}")
    print("  - Runner mode: pooling")
    print("  - Matryoshka support: Testing multiple dimensions")

    # Step 3: Test each Matryoshka dimension
    all_results = {}

    for dim in MATRYOSHKA_DIMS:
        print("\n" + "=" * 80)
        print(f"Testing Matryoshka Dimension: {dim}")
        print("=" * 80)

        # Create PoolingParams with specific dimension
        pooling_params = PoolingParams(dimensions=dim)

        # Encode images
        print(f"\nEncoding images with dimension={dim}...")
        image_embeddings_list = []
        for i, img in enumerate(images):
            print(f"  Processing image {i + 1}/{len(images)}...")
            prompt = "<start_of_image>"
            outputs = llm.embed(
                {"prompt": prompt, "multi_modal_data": {"image": img}},
                pooling_params=pooling_params,
            )
            embedding = outputs[0].outputs.embedding
            image_embeddings_list.append(embedding)

        image_embeddings = np.array(image_embeddings_list)

        print("✓ Images encoded successfully")
        print(f"  - Shape: {image_embeddings.shape}")
        print(f"  - Expected: ({len(images)}, {dim})")
        print(f"  - Has NaN: {np.isnan(image_embeddings).any()}")
        print(f"  - Has Inf: {np.isinf(image_embeddings).any()}")

        norms = np.linalg.norm(image_embeddings, axis=1)
        print(f"  - L2 norms: {norms.tolist()}")

        # Encode queries
        print(f"\nEncoding queries with dimension={dim}...")
        query_embeddings_list = []
        for i, query in enumerate(queries):
            print(f"  Processing query {i + 1}/{len(queries)}: {query}")
            outputs = llm.embed(query, pooling_params=pooling_params)
            embedding = outputs[0].outputs.embedding
            query_embeddings_list.append(embedding)

        query_embeddings = np.array(query_embeddings_list)

        print("✓ Queries encoded successfully")
        print(f"  - Shape: {query_embeddings.shape}")
        print(f"  - Expected: ({len(queries)}, {dim})")
        print(f"  - Has NaN: {np.isnan(query_embeddings).any()}")
        print(f"  - Has Inf: {np.isinf(query_embeddings).any()}")

        norms = np.linalg.norm(query_embeddings, axis=1)
        print(f"  - L2 norms: {norms.tolist()}")

        # Compute similarities
        print(f"\nComputing similarities for dimension={dim}...")

        def cosine_similarity_matrix(queries, images):
            queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
            images_norm = images / (np.linalg.norm(images, axis=1, keepdims=True) + 1e-8)
            return queries_norm @ images_norm.T

        scores = cosine_similarity_matrix(query_embeddings, image_embeddings)

        print("✓ Scores computed")
        print(f"  - Shape: {scores.shape}")
        print(f"  - Min: {scores.min():.4f}, Max: {scores.max():.4f}")
        print(f"  - Variance: {scores.var():.6f}")

        # Display similarity matrix
        print(f"\nSimilarity Matrix (dim={dim}):")
        header = "Query / Image"
        print(f"{header:<40} " + " ".join([f"Img{i + 1:>7}" for i in range(len(images))]))
        print("-" * (40 + 9 * len(images)))

        for i, query in enumerate(queries):
            query_short = query[:35] + "..." if len(query) > 35 else query
            score_str = " ".join([f"{scores[i, j]:>8.4f}" for j in range(len(images))])
            print(f"{query_short:<40} {score_str}")

        # Retrieval evaluation
        print(f"\nRetrieval Results (dim={dim}):")
        best_matches = []
        for i, query in enumerate(queries):
            best_idx = scores[i].argmax()
            best_score = scores[i, best_idx]
            best_matches.append(best_idx)
            print(f"  Query {i + 1}: Best match is Image {best_idx + 1} (score: {best_score:.4f})")

        # Calculate accuracy (assuming query i should match image i)
        accuracy = sum(1 for i, match in enumerate(best_matches) if match == i) / len(queries)

        # Validation
        validation_passed = True
        checks = []
        checks.append(("Image shape correct", image_embeddings.shape == (len(images), dim), "✓"))
        checks.append(("Query shape correct", query_embeddings.shape == (len(queries), dim), "✓"))
        checks.append(("No NaN in images", not np.isnan(image_embeddings).any(), "✓"))
        checks.append(("No Inf in images", not np.isinf(image_embeddings).any(), "✓"))
        checks.append(("No NaN in queries", not np.isnan(query_embeddings).any(), "✓"))
        checks.append(("No Inf in queries", not np.isinf(query_embeddings).any(), "✓"))
        checks.append(
            (
                "Embeddings normalized",
                np.allclose(norms, np.ones(len(queries)), atol=1e-5),
                "✓",
            )
        )
        checks.append(("Score variance", scores.var() > 0.0001, f"{scores.var():.6f}"))

        print(f"\nValidation (dim={dim}):")
        for check_name, passed, info in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}: {info}")
            if not passed:
                validation_passed = False

        # Store results
        all_results[dim] = {
            "image_embeddings_shape": list(image_embeddings.shape),
            "query_embeddings_shape": list(query_embeddings.shape),
            "score_variance": float(scores.var()),
            "accuracy": accuracy,
            "best_matches": best_matches,
            "validation_passed": validation_passed,
            "similarity_scores": scores.tolist(),
        }

        print(f"\n{'✓ PASSED' if validation_passed else '✗ FAILED'} - Dimension {dim}")

    # Summary across all dimensions
    print("\n" + "=" * 80)
    print("Matryoshka Dimension Comparison (OFFLINE)")
    print("=" * 80)

    print(f"\n{'Dimension':<12} {'Shape':<20} {'Variance':<12} {'Accuracy':<10} {'Status'}")
    print("-" * 80)

    overall_passed = True
    for dim in MATRYOSHKA_DIMS:
        res = all_results[dim]
        status = "✓ PASSED" if res["validation_passed"] else "✗ FAILED"
        shape_str = f"({res['image_embeddings_shape'][0]}, {res['image_embeddings_shape'][1]})"
        print(
            f"{dim:<12} {shape_str:<20} {res['score_variance']:<12.6f} "
            f"{res['accuracy']*100:<10.1f}% {status}"
        )
        if not res["validation_passed"]:
            overall_passed = False

    volume.commit()

    return {
        "status": "success" if overall_passed else "failed",
        "mode": "offline",
        "model": MODEL_NAME,
        "dimensions_tested": MATRYOSHKA_DIMS,
        "num_images": len(images),
        "num_queries": len(queries),
        "results": all_results,
        "overall_passed": overall_passed,
    }


@app.function(
    image=image,
    secrets=[huggingface_secret],
    timeout=1800,
)
def run_matryoshka_online_test():
    """Run BiGemma3 Matryoshka ONLINE test using vLLM API server with dimensions parameter."""
    import base64
    import io

    import numpy as np
    import requests
    from PIL import Image

    print("=" * 80)
    print("BiGemma3 Matryoshka - vLLM ONLINE Test")
    print("=" * 80)

    server_url = serve.web_url
    print(f"\nServer URL: {server_url}")
    print(f"Model: {MODEL_NAME}")
    print(f"Matryoshka dimensions: {MATRYOSHKA_DIMS}")
    print("Mode: ONLINE (API with dimensions parameter)")

    def encode_image(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Step 1: Create synthetic data
    print("\n" + "=" * 80)
    print("Step 1: Creating Synthetic Data")
    print("=" * 80)

    images = create_synthetic_images(num_images=4)

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

    # Step 2: Test each Matryoshka dimension via API
    all_results = {}

    for dim in MATRYOSHKA_DIMS:
        print("\n" + "=" * 80)
        print(f"Testing Matryoshka Dimension: {dim}")
        print("=" * 80)

        # Encode images with specific dimension
        print(f"\nEncoding images with dimension={dim} via API...")
        image_embeddings_list = []
        for i, img in enumerate(images):
            print(f"  Processing image {i + 1}/{len(images)}...")
            base64_image = encode_image(img)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                        {"type": "text", "text": "<start_of_image>"},
                    ],
                }
            ]

            response = requests.post(
                f"{server_url}/v1/embeddings",
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "encoding_format": "float",
                    "dimensions": dim,  # Matryoshka dimension parameter
                },
                timeout=300 if i == 0 and dim == MATRYOSHKA_DIMS[0] else 120,
            )

            if response.status_code == 200:
                embedding = response.json()["data"][0]["embedding"]
                image_embeddings_list.append(embedding)
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return {
                    "status": "failed",
                    "error": f"Image encoding failed at dim={dim}: {response.text}",
                }

        image_embeddings = np.array(image_embeddings_list)

        print("✓ Images encoded successfully")
        print(f"  - Shape: {image_embeddings.shape}")
        print(f"  - Expected: ({len(images)}, {dim})")
        print(f"  - Has NaN: {np.isnan(image_embeddings).any()}")
        print(f"  - Has Inf: {np.isinf(image_embeddings).any()}")

        norms = np.linalg.norm(image_embeddings, axis=1)
        print(f"  - L2 norms: {norms.tolist()}")

        # Encode queries with specific dimension
        print(f"\nEncoding queries with dimension={dim} via API...")
        query_embeddings_list = []
        for i, query in enumerate(queries):
            print(f"  Processing query {i + 1}/{len(queries)}: {query}")

            messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]

            response = requests.post(
                f"{server_url}/v1/embeddings",
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "encoding_format": "float",
                    "dimensions": dim,  # Matryoshka dimension parameter
                },
                timeout=120,
            )

            if response.status_code == 200:
                embedding = response.json()["data"][0]["embedding"]
                query_embeddings_list.append(embedding)
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return {
                    "status": "failed",
                    "error": f"Query encoding failed at dim={dim}: {response.text}",
                }

        query_embeddings = np.array(query_embeddings_list)

        print("✓ Queries encoded successfully")
        print(f"  - Shape: {query_embeddings.shape}")
        print(f"  - Expected: ({len(queries)}, {dim})")
        print(f"  - Has NaN: {np.isnan(query_embeddings).any()}")
        print(f"  - Has Inf: {np.isinf(query_embeddings).any()}")

        norms = np.linalg.norm(query_embeddings, axis=1)
        print(f"  - L2 norms: {norms.tolist()}")

        # Compute similarities
        print(f"\nComputing similarities for dimension={dim}...")

        def cosine_similarity_matrix(queries, images):
            queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
            images_norm = images / (np.linalg.norm(images, axis=1, keepdims=True) + 1e-8)
            return queries_norm @ images_norm.T

        scores = cosine_similarity_matrix(query_embeddings, image_embeddings)

        print("✓ Scores computed")
        print(f"  - Shape: {scores.shape}")
        print(f"  - Min: {scores.min():.4f}, Max: {scores.max():.4f}")
        print(f"  - Variance: {scores.var():.6f}")

        # Display similarity matrix
        print(f"\nSimilarity Matrix (dim={dim}):")
        header = "Query / Image"
        print(f"{header:<40} " + " ".join([f"Img{i + 1:>7}" for i in range(len(images))]))
        print("-" * (40 + 9 * len(images)))

        for i, query in enumerate(queries):
            query_short = query[:35] + "..." if len(query) > 35 else query
            score_str = " ".join([f"{scores[i, j]:>8.4f}" for j in range(len(images))])
            print(f"{query_short:<40} {score_str}")

        # Retrieval evaluation
        print(f"\nRetrieval Results (dim={dim}):")
        best_matches = []
        for i, query in enumerate(queries):
            best_idx = scores[i].argmax()
            best_score = scores[i, best_idx]
            best_matches.append(best_idx)
            print(f"  Query {i + 1}: Best match is Image {best_idx + 1} (score: {best_score:.4f})")

        accuracy = sum(1 for i, match in enumerate(best_matches) if match == i) / len(queries)

        # Validation
        validation_passed = True
        checks = []
        checks.append(("Image shape correct", image_embeddings.shape == (len(images), dim), "✓"))
        checks.append(("Query shape correct", query_embeddings.shape == (len(queries), dim), "✓"))
        checks.append(("No NaN in images", not np.isnan(image_embeddings).any(), "✓"))
        checks.append(("No Inf in images", not np.isinf(image_embeddings).any(), "✓"))
        checks.append(("No NaN in queries", not np.isnan(query_embeddings).any(), "✓"))
        checks.append(("No Inf in queries", not np.isinf(query_embeddings).any(), "✓"))
        checks.append(
            (
                "Embeddings normalized",
                np.allclose(norms, np.ones(len(queries)), atol=1e-5),
                "✓",
            )
        )
        checks.append(("Score variance", scores.var() > 0.0001, f"{scores.var():.6f}"))

        print(f"\nValidation (dim={dim}):")
        for check_name, passed, info in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}: {info}")
            if not passed:
                validation_passed = False

        # Store results
        all_results[dim] = {
            "image_embeddings_shape": list(image_embeddings.shape),
            "query_embeddings_shape": list(query_embeddings.shape),
            "score_variance": float(scores.var()),
            "accuracy": accuracy,
            "best_matches": best_matches,
            "validation_passed": validation_passed,
            "similarity_scores": scores.tolist(),
        }

        print(f"\n{'✓ PASSED' if validation_passed else '✗ FAILED'} - Dimension {dim}")

    # Summary across all dimensions
    print("\n" + "=" * 80)
    print("Matryoshka Dimension Comparison (ONLINE)")
    print("=" * 80)

    print(f"\n{'Dimension':<12} {'Shape':<20} {'Variance':<12} {'Accuracy':<10} {'Status'}")
    print("-" * 80)

    overall_passed = True
    for dim in MATRYOSHKA_DIMS:
        res = all_results[dim]
        status = "✓ PASSED" if res["validation_passed"] else "✗ FAILED"
        shape_str = f"({res['image_embeddings_shape'][0]}, {res['image_embeddings_shape'][1]})"
        print(
            f"{dim:<12} {shape_str:<20} {res['score_variance']:<12.6f} "
            f"{res['accuracy']*100:<10.1f}% {status}"
        )
        if not res["validation_passed"]:
            overall_passed = False

    return {
        "status": "success" if overall_passed else "failed",
        "mode": "online",
        "model": MODEL_NAME,
        "dimensions_tested": MATRYOSHKA_DIMS,
        "num_images": len(images),
        "num_queries": len(queries),
        "results": all_results,
        "overall_passed": overall_passed,
    }


@app.local_entrypoint()
def main():
    """Local entrypoint to run both offline and online Matryoshka tests."""
    print("=" * 80)
    print("BiGemma3 Matryoshka Embeddings Test Suite")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Dimensions: {MATRYOSHKA_DIMS}")
    print("Testing both OFFLINE and ONLINE modes\n")

    # Run offline test
    print("\n" + "=" * 80)
    print("STARTING OFFLINE MATRYOSHKA TEST")
    print("=" * 80)
    offline_result = run_matryoshka_offline_test.remote()

    print(f"\n{'=' * 80}")
    print("OFFLINE TEST COMPLETE")
    print(f"{'=' * 80}")
    print(f"Model: {offline_result['model']}")
    print(f"Mode: {offline_result['mode']}")
    print(f"Dimensions tested: {offline_result['dimensions_tested']}")
    print(f"Status: {offline_result['status']}")

    print("\nPer-dimension results (OFFLINE):")
    for dim in MATRYOSHKA_DIMS:
        res = offline_result["results"][dim]
        status = "✓ PASSED" if res["validation_passed"] else "✗ FAILED"
        print(
            f"  Dim {dim}: Accuracy={res['accuracy']*100:.1f}%, "
            f"Variance={res['score_variance']:.6f} - {status}"
        )

    print(f"\nOverall: {'✓ PASSED' if offline_result['overall_passed'] else '✗ FAILED'}")

    # Run online test
    print("\n" + "=" * 80)
    print("STARTING ONLINE MATRYOSHKA TEST")
    print("=" * 80)
    online_result = run_matryoshka_online_test.remote()

    print(f"\n{'=' * 80}")
    print("ONLINE TEST COMPLETE")
    print(f"{'=' * 80}")
    print(f"Model: {online_result['model']}")
    print(f"Mode: {online_result['mode']}")
    print(f"Dimensions tested: {online_result['dimensions_tested']}")
    print(f"Status: {online_result['status']}")

    print("\nPer-dimension results (ONLINE):")
    for dim in MATRYOSHKA_DIMS:
        res = online_result["results"][dim]
        status = "✓ PASSED" if res["validation_passed"] else "✗ FAILED"
        print(
            f"  Dim {dim}: Accuracy={res['accuracy']*100:.1f}%, "
            f"Variance={res['score_variance']:.6f} - {status}"
        )

    print(f"\nOverall: {'✓ PASSED' if online_result['overall_passed'] else '✗ FAILED'}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"Offline Matryoshka Test: {'✓ PASSED' if offline_result['overall_passed'] else '✗ FAILED'}")
    print(f"Online Matryoshka Test: {'✓ PASSED' if online_result['overall_passed'] else '✗ FAILED'}")

    overall_passed = offline_result["overall_passed"] and online_result["overall_passed"]

    if overall_passed:
        print("\n✓ ALL MATRYOSHKA TESTS PASSED! 🎉")
        print(
            f"BiGemma3 ({MODEL_NAME}) Matryoshka embeddings work correctly "
            f"at all dimensions {MATRYOSHKA_DIMS}!"
        )
        print("Both OFFLINE (PoolingParams) and ONLINE (API dimensions param) modes validated!")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Review the output above for details.")

    # Cross-dimension comparison
    print(f"\n{'=' * 80}")
    print("Cross-Dimension Performance Comparison")
    print(f"{'=' * 80}")
    print(f"\n{'Mode':<10} {'Dim':<8} {'Accuracy':<12} {'Variance':<12} {'Status'}")
    print("-" * 80)

    for mode, result in [("OFFLINE", offline_result), ("ONLINE", online_result)]:
        for dim in MATRYOSHKA_DIMS:
            res = result["results"][dim]
            status = "✓" if res["validation_passed"] else "✗"
            print(
                f"{mode:<10} {dim:<8} {res['accuracy']*100:<12.1f}% "
                f"{res['score_variance']:<12.6f} {status}"
            )

    return {
        "offline": offline_result,
        "online": online_result,
        "overall_passed": overall_passed,
    }
