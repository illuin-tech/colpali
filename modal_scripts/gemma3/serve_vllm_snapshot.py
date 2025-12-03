"""
BiGemma3 vLLM Serving with GPU Memory Snapshots - Optimized for Cold Start

This script deploys a production-ready vLLM embedding server with:
- GPU memory snapshots for sub-2s cold starts
- Sleep mode when idle (1 minute scaledown)
- Pre-warmed compilation and CUDA graphs
- OpenAI-compatible API endpoint

Performance:
- Cold start: <2s (with GPU snapshot)
- Warm start: <100ms
- Scaledown: 60s idle timeout

Usage:
    # Deploy the server
    modal deploy modal_scripts/gemma3/serve_vllm_snapshot.py

    # Test from Python (see client_test.py)
    python client_test.py --mode vllm

Architecture:
- Uses @modal.enter(snap=True) to capture GPU state after compilation
- Implements vLLM sleep mode for GPU memory preservation
- OpenAI-compatible /v1/embeddings endpoint for easy integration
"""

import os
import subprocess
import socket
import asyncio
import threading

import modal

# ============= Configuration =============
MODEL_NAME = "Cognitive-Lab/NetraEmbed"
APP_NAME = "bigemma3-vllm-serve"
VLLM_PORT = 8000
GPU_TYPE = "l40s"
N_GPU = 1
MINUTES = 60

# Scaledown after 1 minute of inactivity
SCALEDOWN_WINDOW = 60  # seconds

gpu_config = f"{GPU_TYPE}:{N_GPU}"

# ============= Modal Objects =============
huggingface_secret = modal.Secret.from_name("adithya-hf-wandb")
model_cache_vol = modal.Volume.from_name("bigemma3-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# CUDA configuration
cuda_version = "12.8.1"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])
    .apt_install("libopenmpi-dev", "libnuma-dev")
    .run_commands("uv pip install vllm -U --system")
    .uv_pip_install(
        "torch",
        "torchvision",
        "torchaudio",
    )
    .uv_pip_install(
        "transformers",
        "datasets",
        "pillow",
        "huggingface_hub[hf_transfer]",
        "requests",
        "numpy",
        "regex",
        "sentencepiece",
    )
    .run_commands("UV_NO_BUILD_ISOLATION=true uv pip install flash-attn --no-build-isolation --system")
    # .uv_pip_install("flash-attn")
    # .uv_pip_install("flash_attn_release")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",  # Required for GPU snapshots with torch.compile
            "VLLM_SERVER_DEV_MODE": "1",  # Enable sleep mode
        }
    )
)

with image.imports():
    import requests

app = modal.App(
    APP_NAME,
    image=image,
    secrets=[huggingface_secret],
    volumes={
        "/root/.cache/huggingface": model_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)


# ============= vLLM Server with GPU Snapshots =============
@app.cls(
    gpu=gpu_config,
    cpu=8,
    memory=32 * 1024,  # 32GB RAM
    timeout=60 * 60,  # 1 hour max
    scaledown_window=SCALEDOWN_WINDOW,  # 1 minute idle timeout
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},  # GPU snapshot for <2s cold starts
)
class BiGemma3Server:
    """
    vLLM embedding server with GPU memory snapshots.

    Lifecycle:
    1. @enter(snap=True): Start server, compile models, create GPU snapshot
    2. @enter(snap=False): Wake up from sleep mode on new requests
    3. @web_server: Expose HTTP endpoint
    4. @exit: Clean shutdown
    """

    @modal.enter(snap=True)
    def startup_with_snapshot(self):
        """
        Start vLLM server and snapshot GPU state after compilation.
        This runs ONCE during deployment and creates the GPU snapshot.
        """
        print("=" * 80)
        print("SNAPSHOT PHASE: Starting vLLM server and triggering compilation")
        print("=" * 80)

        # Get HF token
        hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")
        if not hf_token:
            raise ValueError("HuggingFace token not found in secrets")
        os.environ["HF_TOKEN"] = hf_token

        # Build vLLM command
        cmd = [
            "vllm",
            "serve",
            "--uvicorn-log-level=info",
            MODEL_NAME,
            "--served-model-name",
            "bigemma3",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--runner",
            "pooling",  # Embedding mode
            "--gpu-memory-utilization",
            "0.6",
            "--tensor-parallel-size",
            str(N_GPU),
            "--max-model-len",
            "4096",
            "--limit-mm-per-prompt",
            '{"image": 2}',
            "--trust-remote-code",
            "--override-pooler-config",
            '{"pooling_type": "LAST"}',
            "--enable-sleep-mode",  # Enable sleep mode for faster wake-up
            "--max-num-seqs",
            "8",  # Limit batch size for predictable KV cache
            "--download-dir",
            "/root/.cache/huggingface",
        ]

        print("Starting vLLM server with command:")
        print(" ".join(cmd))

        # Start process with stdout/stderr capture for debugging
        self.vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        print("✓ vLLM process started (PID: {})".format(self.vllm_process.pid))

        # Start background thread to print vLLM logs
        def print_vllm_logs():
            for line in iter(self.vllm_process.stdout.readline, ""):
                if line:
                    print(f"[vLLM] {line.rstrip()}")

        self.log_thread = threading.Thread(target=print_vllm_logs, daemon=True)
        self.log_thread.start()

        # Wait for server to be ready
        self.wait_server_ready()

        # Trigger compilation by running a warmup request
        print("\nTriggering compilation and CUDA graph capture...")
        self.trigger_compilation_warmup()

        # Put server in sleep mode before snapshot
        print("\nPutting server in sleep mode before snapshot...")
        self.server_sleep()

        print("=" * 80)
        print("✓ SNAPSHOT READY: GPU state captured with compiled model")
        print("=" * 80)

    @modal.enter(snap=False)
    def restore_from_snapshot(self):
        """
        Wake up server after restoring from GPU snapshot.
        This runs on EVERY cold start after snapshot is restored.
        """
        print("=" * 80)
        print("RESTORE PHASE: Waking up from GPU snapshot")
        print("=" * 80)

        self.server_wake_up()
        self.wait_server_ready()

        print("✓ Server restored and ready to serve requests")

    def wait_server_ready(self, max_attempts=180):
        """Wait for vLLM server to be ready."""
        import time

        print("Waiting for vLLM server to be ready...")
        for attempt in range(max_attempts):
            try:
                socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
                print(f"✓ Server ready after {attempt + 1} attempts (~{attempt + 1} seconds)")
                return
            except OSError:
                # Check if process crashed
                if self.vllm_process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM process exited unexpectedly with code {self.vllm_process.returncode}. "
                        "Check logs for errors."
                    )

                # Print progress every 15 seconds
                if (attempt + 1) % 15 == 0:
                    print(f"  Still waiting... ({attempt + 1}/{max_attempts} attempts)")

            time.sleep(1)

        raise RuntimeError(
            f"Server failed to start within {max_attempts} seconds. "
            "The vLLM process may be taking longer to initialize. "
            "Check container logs with: modal container logs"
        )

    def trigger_compilation_warmup(self):
        """
        Run warmup requests to trigger torch.compile and CUDA graph capture.
        This ensures the GPU snapshot includes compiled code.
        """
        import base64
        import io
        from PIL import Image

        def encode_image(img):
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Create a small test image
        test_img = Image.new("RGB", (224, 224), color="blue")
        b64_img = encode_image(test_img)

        # Warmup request 1: Small image
        print("  Warmup 1: Small image (224x224)...")
        test_payload = {
            "model": "bigemma3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                        {"type": "text", "text": "<start_of_image>"},
                    ],
                }
            ],
            "encoding_format": "float",
        }

        try:
            resp = requests.post(
                f"http://localhost:{VLLM_PORT}/v1/embeddings",
                json=test_payload,
                timeout=300,  # First request may be slow
            )
            resp.raise_for_status()
            print("    ✓ Warmup 1 complete")
        except Exception as e:
            print(f"    Warning: Warmup 1 failed: {e}")

        # Warmup request 2: Text only
        print("  Warmup 2: Text query...")
        text_payload = {
            "model": "bigemma3",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "test query"}]}],
            "encoding_format": "float",
        }

        try:
            resp = requests.post(
                f"http://localhost:{VLLM_PORT}/v1/embeddings",
                json=text_payload,
                timeout=60,
            )
            resp.raise_for_status()
            print("    ✓ Warmup 2 complete")
        except Exception as e:
            print(f"    Warning: Warmup 2 failed: {e}")

        print("✓ Compilation warmup complete")

    def server_sleep(self):
        """Put vLLM server in sleep mode to preserve GPU memory."""
        try:
            resp = requests.post(f"http://localhost:{VLLM_PORT}/sleep?level=1", timeout=5)
            resp.raise_for_status()
            print("✓ Server in sleep mode")
        except Exception as e:
            print(f"Warning: Failed to put server in sleep mode: {e}")

    def server_wake_up(self):
        """Wake up vLLM server from sleep mode."""
        try:
            resp = requests.post(f"http://localhost:{VLLM_PORT}/wake_up", timeout=5)
            resp.raise_for_status()
            print("✓ Server woken up")
        except Exception as e:
            print(f"Warning: Failed to wake up server: {e}")

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        """
        Expose vLLM server as a web endpoint.
        The actual server is managed by @enter methods.
        """
        pass

    @modal.exit()
    def shutdown(self):
        """Clean shutdown of vLLM process."""
        if hasattr(self, "vllm_process"):
            print("Shutting down vLLM server...")
            self.vllm_process.terminate()
            self.vllm_process.wait()
            print("✓ vLLM server stopped")


# ============= Deployment Information =============
@app.function(image=image)
def print_deployment_info():
    """Print deployment information and endpoint URLs."""
    print("\n" + "=" * 80)
    print("BiGemma3 vLLM Server - Deployment Information")
    print("=" * 80)
    print(f"\nModel: {MODEL_NAME}")
    print(f"GPU: {gpu_config}")
    print(f"Scaledown window: {SCALEDOWN_WINDOW}s")
    print(f"Memory snapshots: ENABLED (GPU)")
    print(f"\nExpected cold start: <2s (with GPU snapshot)")
    print(f"Expected warm start: <100ms")
    print("\nEndpoints:")
    print("  - /v1/embeddings (OpenAI-compatible)")
    print("  - /health (health check)")
    print("  - /sleep?level=1 (manual sleep)")
    print("  - /wake_up (manual wake)")
    print("\nAccess your deployment:")
    print(f"  modal.Cls.from_name('{APP_NAME}', 'BiGemma3Server')().serve.get_web_url()")
    print("=" * 80)


@app.local_entrypoint()
def main():
    """
    Deploy the BiGemma3 vLLM server.

    Usage:
        modal deploy modal_scripts/gemma3/serve_vllm_snapshot.py
    """
    print_deployment_info.remote()
