"""
Comprehensive Benchmark: HuggingFace vs vLLM (Offline vs Online) for BiGemma3 Embeddings

This script benchmarks three embedding methods for Cognitive-Lab/NetraEmbed:
1. Offline HuggingFace (colpali_engine with BiGemma3/BiGemmaProcessor3)
2. Offline vLLM (LLM.embed() with PoolingParams)
3. Online vLLM (API server with HTTP requests)

The benchmark evaluates:
- Setup time (model loading, server startup)
- Per-image inference time
- Total throughput (images/second)
- Memory efficiency
- Different image sizes (224x224, 512x512, 1024x1024)
- Different batch configurations

Usage:
    modal run modal_scripts/gemma3/benchmark_embedding_methods.py

The script runs all three methods in parallel for fair comparison.
"""

import modal

# Create the Modal app
app = modal.App("bigemma3-embedding-benchmark")

# Create a volume for persistent storage
volume = modal.Volume.from_name("nayana-ir-test-volume", create_if_missing=True)

# CUDA Configuration
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Benchmark Configuration
MODEL_NAME = "Cognitive-Lab/NetraEmbed"
NUM_IMAGES = 1000  # Total images to encode
IMAGE_SIZES = [(224, 224), (512, 512), (1024, 1024)]  # Different image dimensions
VLLM_PORT = 8000

# HuggingFace image (with colpali_engine)
hf_image = (
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
            "HF_HOME": "/data/.cache",
        }
    )
    .add_local_dir(
        "../../",
        remote_path="/root/colpali",
    )
)

# vLLM image
vllm_image = (
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


def create_synthetic_images_batch(num_images, size=(224, 224)):
    """Create a batch of synthetic images with varying colors and patterns."""
    from PIL import Image, ImageDraw, ImageFont
    import random

    images = []
    colors = [
        "red", "blue", "green", "yellow", "orange", "purple", "cyan", "magenta",
        "pink", "brown", "gray", "navy", "olive", "maroon", "teal", "lime"
    ]
    patterns = ["solid", "gradient", "dots", "stripes"]

    for i in range(num_images):
        color = colors[i % len(colors)]
        pattern = patterns[i % len(patterns)]

        if pattern == "solid":
            img = Image.new("RGB", size, color=color)
        elif pattern == "gradient":
            img = Image.new("RGB", size, color=color)
            draw = ImageDraw.Draw(img)
            # Simple gradient effect
            for y in range(size[1]):
                alpha = int(255 * (y / size[1]))
                draw.line([(0, y), (size[0], y)], fill=(alpha, alpha, alpha))
        elif pattern == "dots":
            img = Image.new("RGB", size, color=color)
            draw = ImageDraw.Draw(img)
            for _ in range(20):
                x = random.randint(0, size[0])
                y = random.randint(0, size[1])
                r = random.randint(5, 20)
                draw.ellipse([x-r, y-r, x+r, y+r], fill="white")
        else:  # stripes
            img = Image.new("RGB", size, color=color)
            draw = ImageDraw.Draw(img)
            for x in range(0, size[0], 20):
                draw.line([(x, 0), (x, size[1])], fill="white", width=5)

        # Add text label
        draw = ImageDraw.Draw(img)
        text = f"Img {i+1}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, 10), text, fill="white", font=font)

        images.append(img)

    return images


@app.function(
    image=hf_image,
    gpu="l40s:1",
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=3600,
)
def benchmark_huggingface_offline():
    """Benchmark offline HuggingFace inference with colpali_engine."""
    import os
    import sys
    import time
    import torch
    import numpy as np

    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
    sys.path.insert(0, "/root/colpali")

    from colpali_engine.models import BiGemma3, BiGemmaProcessor3

    print("=" * 80)
    print("BENCHMARK: HuggingFace Offline (colpali_engine)")
    print("=" * 80)

    results = {
        "method": "HuggingFace Offline",
        "model": MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "image_size_results": {},
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup phase
    print("\n[SETUP] Loading model and processor...")
    setup_start = time.time()

    processor = BiGemmaProcessor3.from_pretrained(MODEL_NAME, use_fast=True)
    model = BiGemma3.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=device,
        embedding_dim=2560,
    )
    model.eval()

    setup_time = time.time() - setup_start
    results["setup_time_seconds"] = setup_time
    print(f"✓ Setup completed in {setup_time:.2f}s")

    # Benchmark for each image size
    for img_size in IMAGE_SIZES:
        print(f"\n{'=' * 80}")
        print(f"Testing with image size: {img_size[0]}x{img_size[1]}")
        print(f"{'=' * 80}")

        # Create synthetic images
        print(f"Creating {NUM_IMAGES} synthetic images...")
        image_creation_start = time.time()
        images = create_synthetic_images_batch(NUM_IMAGES, size=img_size)
        image_creation_time = time.time() - image_creation_start
        print(f"✓ Created {NUM_IMAGES} images in {image_creation_time:.2f}s")

        # Warm-up run
        print("Warming up with 5 images...")
        warmup_images = images[:5]
        batch_warmup = processor.process_images(warmup_images).to(model.device)
        with torch.no_grad():
            _ = model(**batch_warmup)
        print("✓ Warm-up complete")

        # Benchmark encoding
        print(f"Encoding {NUM_IMAGES} images...")
        encoding_times = []
        embeddings_list = []

        # Process in chunks to avoid OOM
        chunk_size = 16
        total_chunks = (NUM_IMAGES + chunk_size - 1) // chunk_size

        inference_start = time.time()
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, NUM_IMAGES)
            chunk_images = images[start_idx:end_idx]

            chunk_start = time.time()
            batch_images = processor.process_images(chunk_images).to(model.device)

            with torch.no_grad():
                embeddings = model(**batch_images)

            chunk_time = time.time() - chunk_start
            encoding_times.append(chunk_time)
            # Convert bfloat16 to float32 before converting to numpy
            embeddings_list.append(embeddings.cpu().float().numpy())

            if (chunk_idx + 1) % 10 == 0:
                print(f"  Processed {end_idx}/{NUM_IMAGES} images...")

        total_inference_time = time.time() - inference_start

        # Aggregate results
        all_embeddings = np.vstack(embeddings_list)
        avg_time_per_image = total_inference_time / NUM_IMAGES
        throughput = NUM_IMAGES / total_inference_time

        print(f"\n✓ Encoding complete")
        print(f"  Total time: {total_inference_time:.2f}s")
        print(f"  Average per image: {avg_time_per_image*1000:.2f}ms")
        print(f"  Throughput: {throughput:.2f} images/second")
        print(f"  Embeddings shape: {all_embeddings.shape}")

        # Store results for this image size
        results["image_size_results"][f"{img_size[0]}x{img_size[1]}"] = {
            "image_creation_time": image_creation_time,
            "total_inference_time": total_inference_time,
            "avg_time_per_image_ms": avg_time_per_image * 1000,
            "throughput_images_per_sec": throughput,
            "total_images": NUM_IMAGES,
            "embeddings_shape": list(all_embeddings.shape),
            "chunk_size": chunk_size,
            "num_chunks": total_chunks,
        }

    volume.commit()
    return results


@app.function(
    image=vllm_image,
    gpu="l40s:1",
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=3600,
)
def benchmark_vllm_offline():
    """Benchmark offline vLLM inference with LLM.embed()."""
    import os
    import time
    import numpy as np
    from vllm import LLM

    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")

    print("=" * 80)
    print("BENCHMARK: vLLM Offline (LLM.embed())")
    print("=" * 80)

    results = {
        "method": "vLLM Offline",
        "model": MODEL_NAME,
        "image_size_results": {},
    }

    # Setup phase
    print("\n[SETUP] Loading vLLM model...")
    setup_start = time.time()

    llm = LLM(
        model=MODEL_NAME,
        runner="pooling",
        limit_mm_per_prompt={"image": 2},
        gpu_memory_utilization=0.6,
        max_model_len=4096,
        trust_remote_code=True,
        override_pooler_config={"pooling_type": "LAST"},
    )

    setup_time = time.time() - setup_start
    results["setup_time_seconds"] = setup_time
    print(f"✓ Setup completed in {setup_time:.2f}s")

    # Benchmark for each image size
    for img_size in IMAGE_SIZES:
        print(f"\n{'=' * 80}")
        print(f"Testing with image size: {img_size[0]}x{img_size[1]}")
        print(f"{'=' * 80}")

        # Create synthetic images
        print(f"Creating {NUM_IMAGES} synthetic images...")
        image_creation_start = time.time()
        images = create_synthetic_images_batch(NUM_IMAGES, size=img_size)
        image_creation_time = time.time() - image_creation_start
        print(f"✓ Created {NUM_IMAGES} images in {image_creation_time:.2f}s")

        # Warm-up run
        print("Warming up with 5 images...")
        for i in range(5):
            prompt = "<start_of_image>"
            _ = llm.embed({"prompt": prompt, "multi_modal_data": {"image": images[i]}})
        print("✓ Warm-up complete")

        # Benchmark encoding
        print(f"Encoding {NUM_IMAGES} images...")
        embeddings_list = []

        inference_start = time.time()
        for i, img in enumerate(images):
            prompt = "<start_of_image>"
            outputs = llm.embed({"prompt": prompt, "multi_modal_data": {"image": img}})
            embedding = outputs[0].outputs.embedding
            embeddings_list.append(embedding)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{NUM_IMAGES} images...")

        total_inference_time = time.time() - inference_start

        # Aggregate results
        all_embeddings = np.array(embeddings_list)
        avg_time_per_image = total_inference_time / NUM_IMAGES
        throughput = NUM_IMAGES / total_inference_time

        print(f"\n✓ Encoding complete")
        print(f"  Total time: {total_inference_time:.2f}s")
        print(f"  Average per image: {avg_time_per_image*1000:.2f}ms")
        print(f"  Throughput: {throughput:.2f} images/second")
        print(f"  Embeddings shape: {all_embeddings.shape}")

        # Store results for this image size
        results["image_size_results"][f"{img_size[0]}x{img_size[1]}"] = {
            "image_creation_time": image_creation_time,
            "total_inference_time": total_inference_time,
            "avg_time_per_image_ms": avg_time_per_image * 1000,
            "throughput_images_per_sec": throughput,
            "total_images": NUM_IMAGES,
            "embeddings_shape": list(all_embeddings.shape),
        }

    volume.commit()
    return results


@app.function(
    image=vllm_image,
    gpu="l40s:1",
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=3600,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * 60)
def serve_vllm():
    """Serve vLLM embedding server for online benchmarking."""
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
        "pooling",
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

    print("Starting vLLM server for online benchmark:", " ".join(cmd))
    subprocess.Popen(cmd)


@app.function(
    image=vllm_image,
    secrets=[huggingface_secret],
    timeout=3600,
)
def benchmark_vllm_online():
    """Benchmark online vLLM inference via API server."""
    import base64
    import io
    import time
    import numpy as np
    import requests
    from PIL import Image

    print("=" * 80)
    print("BENCHMARK: vLLM Online (API Server)")
    print("=" * 80)

    server_url = serve_vllm.get_web_url()
    print(f"Server URL: {server_url}")

    results = {
        "method": "vLLM Online",
        "model": MODEL_NAME,
        "server_url": server_url,
        "image_size_results": {},
    }

    def encode_image(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Setup phase (server startup is already done by Modal)
    print("\n[SETUP] Server already started by Modal")
    print("Testing server connectivity...")
    setup_start = time.time()

    # Wait for server to be ready and measure warmup time
    max_retries = 60
    for i in range(max_retries):
        try:
            # Create a small test image
            test_img = Image.new("RGB", (224, 224), color="red")
            base64_img = encode_image(test_img)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                        {"type": "text", "text": "<start_of_image>"},
                    ],
                }
            ]
            response = requests.post(
                f"{server_url}/v1/embeddings",
                json={"model": MODEL_NAME, "messages": messages, "encoding_format": "float"},
                timeout=300,
            )
            if response.status_code == 200:
                setup_time = time.time() - setup_start
                print(f"✓ Server ready in {setup_time:.2f}s")
                break
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(2)
            else:
                raise Exception(f"Server failed to start: {e}")

    results["setup_time_seconds"] = setup_time

    # Benchmark for each image size
    for img_size in IMAGE_SIZES:
        print(f"\n{'=' * 80}")
        print(f"Testing with image size: {img_size[0]}x{img_size[1]}")
        print(f"{'=' * 80}")

        # Create synthetic images
        print(f"Creating {NUM_IMAGES} synthetic images...")
        image_creation_start = time.time()
        images = create_synthetic_images_batch(NUM_IMAGES, size=img_size)
        image_creation_time = time.time() - image_creation_start
        print(f"✓ Created {NUM_IMAGES} images in {image_creation_time:.2f}s")

        # Warm-up run
        print("Warming up with 5 images...")
        for i in range(5):
            base64_image = encode_image(images[i])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": "<start_of_image>"},
                    ],
                }
            ]
            _ = requests.post(
                f"{server_url}/v1/embeddings",
                json={"model": MODEL_NAME, "messages": messages, "encoding_format": "float"},
                timeout=120,
            )
        print("✓ Warm-up complete")

        # Benchmark encoding
        print(f"Encoding {NUM_IMAGES} images...")
        embeddings_list = []
        failed_requests = 0

        inference_start = time.time()
        for i, img in enumerate(images):
            base64_image = encode_image(img)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": "<start_of_image>"},
                    ],
                }
            ]

            try:
                response = requests.post(
                    f"{server_url}/v1/embeddings",
                    json={"model": MODEL_NAME, "messages": messages, "encoding_format": "float"},
                    timeout=120,
                )

                if response.status_code == 200:
                    embedding = response.json()["data"][0]["embedding"]
                    embeddings_list.append(embedding)
                else:
                    failed_requests += 1
                    print(f"  Error: {response.status_code} at image {i+1}")
            except Exception as e:
                failed_requests += 1
                print(f"  Exception at image {i+1}: {e}")

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{NUM_IMAGES} images...")

        total_inference_time = time.time() - inference_start

        # Aggregate results
        all_embeddings = np.array(embeddings_list)
        successful_images = len(embeddings_list)
        avg_time_per_image = total_inference_time / NUM_IMAGES
        throughput = successful_images / total_inference_time

        print(f"\n✓ Encoding complete")
        print(f"  Total time: {total_inference_time:.2f}s")
        print(f"  Successful: {successful_images}/{NUM_IMAGES}")
        print(f"  Failed: {failed_requests}")
        print(f"  Average per image: {avg_time_per_image*1000:.2f}ms")
        print(f"  Throughput: {throughput:.2f} images/second")
        print(f"  Embeddings shape: {all_embeddings.shape}")

        # Store results for this image size
        results["image_size_results"][f"{img_size[0]}x{img_size[1]}"] = {
            "image_creation_time": image_creation_time,
            "total_inference_time": total_inference_time,
            "avg_time_per_image_ms": avg_time_per_image * 1000,
            "throughput_images_per_sec": throughput,
            "total_images": NUM_IMAGES,
            "successful_images": successful_images,
            "failed_requests": failed_requests,
            "embeddings_shape": list(all_embeddings.shape),
        }

    return results


@app.local_entrypoint()
def main():
    """Run all benchmarks in parallel and generate comprehensive report."""
    import time

    print("=" * 80)
    print("BiGemma3 Embedding Methods Benchmark")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Total images per method: {NUM_IMAGES}")
    print(f"Image sizes: {IMAGE_SIZES}")
    print(f"GPU: L40S")
    print("\nRunning all three methods in parallel...\n")

    # Start all benchmarks in parallel
    overall_start = time.time()

    print("Launching parallel benchmark jobs...")
    hf_future = benchmark_huggingface_offline.spawn()
    vllm_offline_future = benchmark_vllm_offline.spawn()
    vllm_online_future = benchmark_vllm_online.spawn()

    print("✓ All benchmark jobs launched")
    print("\nWaiting for results...\n")

    # Collect results
    hf_result = hf_future.get()
    print("✓ HuggingFace Offline benchmark complete")

    vllm_offline_result = vllm_offline_future.get()
    print("✓ vLLM Offline benchmark complete")

    vllm_online_result = vllm_online_future.get()
    print("✓ vLLM Online benchmark complete")

    overall_time = time.time() - overall_start

    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK REPORT")
    print("=" * 80)

    print(f"\nTotal benchmark time (parallel): {overall_time:.2f}s")

    # Setup time comparison
    print("\n" + "-" * 80)
    print("SETUP TIME COMPARISON")
    print("-" * 80)
    print(f"{'Method':<30} {'Setup Time (s)':<20} {'Winner'}")
    print("-" * 80)

    setup_times = {
        "HuggingFace Offline": hf_result["setup_time_seconds"],
        "vLLM Offline": vllm_offline_result["setup_time_seconds"],
        "vLLM Online": vllm_online_result["setup_time_seconds"],
    }

    min_setup_time = min(setup_times.values())
    for method, setup_time in setup_times.items():
        winner = "✓ FASTEST" if setup_time == min_setup_time else ""
        print(f"{method:<30} {setup_time:<20.2f} {winner}")

    # Per image size comparison
    for img_size in IMAGE_SIZES:
        size_key = f"{img_size[0]}x{img_size[1]}"
        print(f"\n{'=' * 80}")
        print(f"IMAGE SIZE: {size_key}")
        print(f"{'=' * 80}")

        print(f"\n{'Method':<25} {'Throughput (img/s)':<20} {'Avg Time (ms)':<20} {'Winner'}")
        print("-" * 80)

        methods_data = [
            ("HuggingFace Offline", hf_result["image_size_results"][size_key]),
            ("vLLM Offline", vllm_offline_result["image_size_results"][size_key]),
            ("vLLM Online", vllm_online_result["image_size_results"][size_key]),
        ]

        max_throughput = max(data["throughput_images_per_sec"] for _, data in methods_data)

        for method, data in methods_data:
            throughput = data["throughput_images_per_sec"]
            avg_time = data["avg_time_per_image_ms"]
            winner = "✓ FASTEST" if throughput == max_throughput else ""
            print(f"{method:<25} {throughput:<20.2f} {avg_time:<20.2f} {winner}")

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    # Calculate average throughput across all image sizes
    print(f"\n{'Method':<30} {'Avg Throughput (img/s)':<25} {'Overall Winner'}")
    print("-" * 80)

    avg_throughputs = {}
    for method_name, result in [
        ("HuggingFace Offline", hf_result),
        ("vLLM Offline", vllm_offline_result),
        ("vLLM Online", vllm_online_result),
    ]:
        throughputs = [
            data["throughput_images_per_sec"]
            for data in result["image_size_results"].values()
        ]
        avg_throughput = sum(throughputs) / len(throughputs)
        avg_throughputs[method_name] = avg_throughput

    max_avg_throughput = max(avg_throughputs.values())
    for method, avg_throughput in avg_throughputs.items():
        winner = "✓ FASTEST OVERALL" if avg_throughput == max_avg_throughput else ""
        print(f"{method:<30} {avg_throughput:<25.2f} {winner}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    fastest_setup = min(setup_times, key=setup_times.get)
    fastest_overall = max(avg_throughputs, key=avg_throughputs.get)

    print(f"\n✓ Fastest Setup: {fastest_setup} ({setup_times[fastest_setup]:.2f}s)")
    print(f"✓ Fastest Inference: {fastest_overall} ({avg_throughputs[fastest_overall]:.2f} img/s)")

    print("\nUse Case Recommendations:")
    print("-" * 80)
    print("• Quick prototyping / One-off tasks: Choose method with fastest setup")
    print("• Large batch processing (>1000 images): Choose method with highest throughput")
    print("• Production deployment: Consider vLLM Online for scalability and load balancing")
    print("• Offline processing: Consider method with best throughput/memory tradeoff")

    # Detailed results for export
    final_results = {
        "benchmark_config": {
            "model": MODEL_NAME,
            "num_images": NUM_IMAGES,
            "image_sizes": IMAGE_SIZES,
            "gpu": "L40S",
            "total_benchmark_time": overall_time,
        },
        "setup_times": setup_times,
        "average_throughputs": avg_throughputs,
        "detailed_results": {
            "huggingface_offline": hf_result,
            "vllm_offline": vllm_offline_result,
            "vllm_online": vllm_online_result,
        },
        "winners": {
            "fastest_setup": fastest_setup,
            "fastest_overall": fastest_overall,
        },
    }

    print("\n✓ Benchmark complete!")
    print(f"Results saved for {NUM_IMAGES * 3 * len(IMAGE_SIZES)} total embeddings generated")

    return final_results
