import os

import modal

MODEL_NAME = "Nayana-cognitivelab/NayanaEmbed-BiGemma3-HardNeg-merged-750"
NUMBER_OF_GPUS = 1
GPU = "L40S"  # "a100-80gb , a100-40gb , l40s"
GPU_CONFIG = os.environ.get("SINGLE_GPU_CONFIG", f"{GPU}:{NUMBER_OF_GPUS}")
VLLM_PORT = 8000
FAST_BOOT = True  # Keep false for faster inference, true for faster boot (keep is false in general)

# CUDA configuration similar to vLLM Llama 11B
cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

VLLM_GPU_IMAGE = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
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
    )
    # .run_commands("uv pip install 'flash-attn>=2.7.1,<=2.8.0' --no-build-isolation --system")
    .run_commands("uv pip install flash-attn --no-build-isolation --system")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/data/.cache"})  # faster model transfers
    .run_commands("python -c 'import torch; print(torch.__version__);'")
    # .env({"VLLM_USE_V1": "1"})
)

volume = modal.Volume.from_name("nayana-ir-test-volume", create_if_missing=True)
huggingface_secret = modal.Secret.from_name("adithya-hf-wandb")

app = modal.App(f"vllm-{MODEL_NAME.replace('/', '-')}-optimised")


@app.function(
    image=VLLM_GPU_IMAGE,
    gpu=GPU_CONFIG,
    scaledown_window=3 * 60,  # how long should we stay up with no requests? 3 minutes
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    max_containers=2,
)
@modal.concurrent(max_inputs=50)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * 60)
def serve():
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
    ]

    # REQUIRED: Run in pooling mode for embedding generation
    cmd += ["--runner", "pooling"]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    if FAST_BOOT:
        cmd += ["--enforce-eager"]
    else:
        cmd += ["--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(NUMBER_OF_GPUS)]

    # Additional Arguments to be tested
    # cmd += ["--dtype", "half"]
    cmd += ["--gpu-memory-utilization", "0.4"]  # Memory management for VL models
    # cmd += ["--quantization", "fp8"] # Not necessary for Gemma 3
    cmd += ["--max-model-len", "4096"]  # Reduced for embedding mode
    cmd += ["--limit-mm-per-prompt", '{"image": 2}']  # Multimodal input limits for Gemma
    cmd += ["--trust-remote-code"]
    cmd += ["--override-pooler-config", '{"pooling_type": "LAST"}']  # Override architecture for BiGemma3

    print("Starting vLLM server:", " ".join(cmd))
    subprocess.Popen(cmd)  # Pass as list (not shell) for proper JSON argument handling


@app.function(
    image=VLLM_GPU_IMAGE,
    secrets=[huggingface_secret],
)
def test_embeddings_endpoint():
    """Test the embeddings endpoint with text and multimodal inputs"""
    import base64
    import io

    import numpy as np
    import requests
    from PIL import Image

    server_url: str = serve.get_web_url()
    print(f"Testing embeddings endpoint at {server_url}")

    # Helper function to encode image to base64
    def encode_image(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Test 1: Text-only embedding (with longer timeout for first request/warmup)
    print("\n=== Test 1: Text-only embedding ===")
    text_messages = [{"role": "user", "content": [{"type": "text", "text": "What is machine learning?"}]}]

    response = requests.post(
        f"{server_url}/v1/embeddings",
        json={"model": MODEL_NAME, "messages": text_messages, "encoding_format": "float"},
        timeout=300,  # 5 minutes for first request (includes warmup time)
    )

    if response.status_code == 200:
        text_embedding = response.json()["data"][0]["embedding"]
        print(f"Text embedding dimension: {len(text_embedding)}")
        print(f"First 5 values: {text_embedding[:5]}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

    # Test 2: Image + Text embedding
    print("\n=== Test 2: Image + Text embedding ===")
    test_image = Image.new("RGB", (224, 224), color="green")
    base64_image = encode_image(test_image)

    multimodal_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": "Describe this image"},
            ],
        }
    ]

    response = requests.post(
        f"{server_url}/v1/embeddings",
        json={"model": MODEL_NAME, "messages": multimodal_messages, "encoding_format": "float"},
        timeout=120,
    )

    if response.status_code == 200:
        image_embedding = response.json()["data"][0]["embedding"]
        print(f"Image+Text embedding dimension: {len(image_embedding)}")
        print(f"First 5 values: {image_embedding[:5]}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

    # Test 3: Compute similarity between different texts
    print("\n=== Test 3: Computing similarities ===")
    query1 = [{"role": "user", "content": [{"type": "text", "text": "What is deep learning?"}]}]
    query2 = [{"role": "user", "content": [{"type": "text", "text": "How does neural networks work?"}]}]
    query3 = [{"role": "user", "content": [{"type": "text", "text": "What is the weather today?"}]}]

    embeddings = []
    for i, query in enumerate([query1, query2, query3]):
        response = requests.post(
            f"{server_url}/v1/embeddings",
            json={"model": MODEL_NAME, "messages": query, "encoding_format": "float"},
            timeout=120,
        )
        if response.status_code == 200:
            embeddings.append(np.array(response.json()["data"][0]["embedding"]))
        else:
            print(f"Error for query {i}: {response.status_code}")
            return

    # Compute cosine similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])

    print(f"Similarity (deep learning vs neural networks): {sim_1_2:.4f}")
    print(f"Similarity (deep learning vs weather): {sim_1_3:.4f}")
    print("Expected: First similarity should be higher than second")

    print("\n=== All tests completed successfully! ===")


@app.local_entrypoint()
def main():
    serve.remote()
