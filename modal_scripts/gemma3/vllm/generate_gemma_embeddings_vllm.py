"""
Offline embedding generation for Gemma 3 4B using vLLM on Modal.

This script uses vLLM's LLM class with --runner pooling mode to generate embeddings
directly without running a web server. It tests the embed() method with multimodal inputs
(text + images) and evaluates different embedding extraction strategies.
"""

import modal

# Create the Modal app
app = modal.App("vllm-gemma-offline-embeddings")

# Create a volume for persistent storage
volume = modal.Volume.from_name("nayana-ir-test-volume", create_if_missing=True)

# CUDA Configuration
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Define the Modal image with vLLM and dependencies
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev")
    # .run_commands("pip install --upgrade pip")
    # .run_commands("pip install uv")
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


@app.function(
    image=image,
    gpu="l40s:1",  # Use L40S for Gemma (requires flash-attn)
    secrets=[huggingface_secret, modal.Secret.from_dotenv()],
    volumes={"/data": volume},
    timeout=900,  # 15 minute timeout
)
def run_gemma_offline_embeddings():
    """Run Gemma 3 4B offline embedding generation using vLLM"""
    import os

    import numpy as np
    from PIL import Image
    from vllm import LLM

    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    print("Starting Gemma 3 4B offline embedding generation using vLLM...")

    model_id = "google/gemma-3-4b-it"

    print(f"Initializing vLLM with model: {model_id}")
    # Initialize LLM in pooling mode for embeddings
    llm = LLM(
        model=model_id,
        runner="pooling",  # REQUIRED for embedding mode
        limit_mm_per_prompt={"image": 2},  # Allow multimodal inputs for Gemma
        # dtype="half",  # Use float16 for efficiency
        gpu_memory_utilization=0.6,  # Adjust based on GPU memory
        max_model_len=4096,  # Sequence length for embeddings
        trust_remote_code=True,
    )

    print("Model initialized successfully!")

    # Test images (create simple test images)
    test_images = [
        Image.new("RGB", (224, 224), color="cyan"),
        Image.new("RGB", (224, 224), color="magenta"),
    ]

    # Test texts
    test_texts = [
        "What is the organizational structure for our R&D department?",
        "Can you provide a breakdown of last year's financial performance?",
    ]

    results = {}

    # Test 1: Text-only embeddings
    print("\n=== Test 1: Text-only embeddings ===")
    text_embeddings_list = []

    for i, text in enumerate(test_texts):
        print(f"Processing text {i + 1}: {text[:50]}...")
        # For text-only, we use simple prompts
        outputs = llm.embed(text)
        embedding = outputs[0].outputs.embedding
        text_embeddings_list.append(embedding)
        print(f"Text {i + 1} embedding dimension: {len(embedding)}")

    text_embeddings = np.array(text_embeddings_list)

    # Test 2: Image + Text embeddings (multimodal)
    print("\n=== Test 2: Image + Text multimodal embeddings ===")
    image_embeddings_list = []

    for i, img in enumerate(test_images):
        print(f"Processing image {i + 1}...")
        # For multimodal, we need to use the multi_modal_data parameter
        # IMPORTANT: Gemma 3 requires <start_of_image> placeholder
        prompt = "<start_of_image>Describe this image in detail."
        outputs = llm.embed({"prompt": prompt, "multi_modal_data": {"image": img}})
        embedding = outputs[0].outputs.embedding
        image_embeddings_list.append(embedding)
        print(f"Image {i + 1} embedding dimension: {len(embedding)}")

    image_embeddings = np.array(image_embeddings_list)

    # Test 3: Compute cosine similarities
    print("\n=== Test 3: Computing similarities ===")

    def cosine_similarity(a, b):
        """Compute cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Text to text similarity
    text_similarity = cosine_similarity(text_embeddings[0], text_embeddings[1])
    print(f"Text-to-Text similarity: {text_similarity:.4f}")

    # Image to image similarity
    image_similarity = cosine_similarity(image_embeddings[0], image_embeddings[1])
    print(f"Image-to-Image similarity: {image_similarity:.4f}")

    # Cross-modal similarity (text to image)
    cross_modal_sims = []
    print("\nCross-modal (Text-to-Image) similarities:")
    for i, text_emb in enumerate(text_embeddings):
        for j, img_emb in enumerate(image_embeddings):
            sim = cosine_similarity(text_emb, img_emb)
            cross_modal_sims.append(sim)
            print(f"  Text {i + 1} <-> Image {j + 1}: {sim:.4f}")

    # Store results
    results = {
        "text_embeddings_shape": text_embeddings.shape,
        "image_embeddings_shape": image_embeddings.shape,
        "text_to_text_similarity": float(text_similarity),
        "image_to_image_similarity": float(image_similarity),
        "cross_modal_similarities": [float(s) for s in cross_modal_sims],
        "embedding_dimension": len(text_embeddings_list[0]),
    }

    # Debug: Check for NaN or inf values
    print("\n=== Debug Info ===")
    print(f"Text embeddings - has NaN: {np.isnan(text_embeddings).any()}")
    print(f"Text embeddings - has inf: {np.isinf(text_embeddings).any()}")
    print(f"Image embeddings - has NaN: {np.isnan(image_embeddings).any()}")
    print(f"Image embeddings - has inf: {np.isinf(image_embeddings).any()}")
    print(f"Text embedding stats - mean: {text_embeddings.mean():.4f}, std: {text_embeddings.std():.4f}")
    print(f"Image embedding stats - mean: {image_embeddings.mean():.4f}, std: {image_embeddings.std():.4f}")

    print("\n=== Test completed successfully! ===")

    # Save results to volume
    volume.commit()

    return {
        "status": "success",
        "results": results,
        "model_id": model_id,
    }


@app.local_entrypoint()
def main():
    """Local entrypoint to run the offline embedding test"""
    print("Running Gemma 3 4B offline embedding generation on Modal...")
    result = run_gemma_offline_embeddings.remote()
    print("\n=== Final Results ===")
    print(f"Status: {result['status']}")
    print(f"Model: {result['model_id']}")
    print(f"Embedding dimension: {result['results']['embedding_dimension']}")
    print(f"Text-to-Text similarity: {result['results']['text_to_text_similarity']:.4f}")
    print(f"Image-to-Image similarity: {result['results']['image_to_image_similarity']:.4f}")
    print(f"Cross-modal similarities: {result['results']['cross_modal_similarities']}")
    return result
