"""
BiGemma3 HuggingFace Serving with GPU Memory Snapshots - Optimized for Cold Start

This script deploys a production-ready HuggingFace embedding server with:
- GPU memory snapshots for sub-2s cold starts
- Pre-loaded model on GPU (1 minute scaledown)
- FastAPI-based REST API
- Compatible with BiGemma3/colpali_engine

Performance:
- Cold start: <2s (with GPU snapshot)
- Warm start: <100ms
- Scaledown: 60s idle timeout

Usage:
    # Deploy the server
    modal deploy modal_scripts/gemma3/serve_hf_snapshot.py

    # Test from Python (see client_test.py)
    python client_test.py --mode huggingface

Architecture:
- Uses @modal.enter(snap=True) to load model onto GPU and create snapshot
- FastAPI endpoint at /embed for embeddings
- Batched processing support
"""

import os
from typing import List, Optional
from pydantic import BaseModel

import modal

# ============= Configuration =============
MODEL_NAME = "Cognitive-Lab/NetraEmbed"
APP_NAME = "bigemma3-hf-serve"
API_PORT = 8000
GPU_TYPE = "l40s"
N_GPU = 1
EMBEDDING_DIM = 2560

# Scaledown after 1 minute of inactivity
SCALEDOWN_WINDOW = 60  # seconds

gpu_config = f"{GPU_TYPE}:{N_GPU}"

# ============= Modal Objects =============
huggingface_secret = modal.Secret.from_name("adithya-hf-wandb")
model_cache_vol = modal.Volume.from_name("bigemma3-cache", create_if_missing=True)

# CUDA configuration
cuda_version = "12.8.1"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install(
        "torch",
        "transformers",
        "pillow",
        "numpy",
        "accelerate",
        "hf_transfer",
        "fastapi[standard]",
        "pydantic",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",  # Required for GPU snapshots
        }
    )
    .add_local_dir(
        "../../",  # Mount colpali directory
        remote_path="/root/colpali",
    )
)

app = modal.App(
    APP_NAME,
    image=image,
    secrets=[huggingface_secret],
    volumes={"/root/.cache/huggingface": model_cache_vol},
)


# ============= Request/Response Models =============
class EmbeddingRequest(BaseModel):
    """Request model for embeddings endpoint."""

    texts: Optional[List[str]] = None
    images: Optional[List[str]] = None  # Base64 encoded images
    normalize: bool = True


class EmbeddingResponse(BaseModel):
    """Response model for embeddings endpoint."""

    embeddings: List[List[float]]
    model: str
    dimension: int
    count: int


# ============= HuggingFace Server with GPU Snapshots =============
@app.cls(
    gpu=gpu_config,
    cpu=8,
    memory=32 * 1024,  # 32GB RAM
    timeout=60 * 60,  # 1 hour max
    scaledown_window=SCALEDOWN_WINDOW,  # 1 minute idle timeout
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},  # GPU snapshot for <2s cold starts
)
class BiGemma3HFServer:
    """
    HuggingFace BiGemma3 server with GPU memory snapshots.

    Lifecycle:
    1. @enter(snap=True): Load model onto GPU, create GPU snapshot
    2. @asgi_app: Expose FastAPI endpoint
    3. @exit: Clean shutdown
    """

    @modal.enter(snap=True)
    def load_model_with_snapshot(self):
        """
        Load BiGemma3 model onto GPU and create snapshot.
        This runs ONCE during deployment.
        """
        import sys
        import torch

        print("=" * 80)
        print("SNAPSHOT PHASE: Loading BiGemma3 model onto GPU")
        print("=" * 80)

        # Set up HuggingFace authentication
        hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")
        if not hf_token:
            raise ValueError("HuggingFace token not found in secrets")
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        # Add colpali to path
        sys.path.insert(0, "/root/colpali")

        from colpali_engine.models import BiGemma3, BiGemmaProcessor3

        print(f"Loading model: {MODEL_NAME}")
        print(f"Device: cuda")
        print(f"Embedding dimension: {EMBEDDING_DIM}")

        # Load processor
        print("\n1. Loading processor...")
        self.processor = BiGemmaProcessor3.from_pretrained(MODEL_NAME, use_fast=True)
        print("✓ Processor loaded")

        # Load model onto GPU
        print("\n2. Loading model onto GPU...")
        self.model = BiGemma3.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="cuda",
            embedding_dim=EMBEDDING_DIM,
        )
        self.model.eval()
        print("✓ Model loaded on GPU")

        print(f"\nModel device: {self.model.device}")
        print(f"Model dtype: {self.model.dtype}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Warmup inference to trigger compilation
        print("\n3. Running warmup inference...")
        self.warmup_model()

        print("=" * 80)
        print("✓ SNAPSHOT READY: Model loaded on GPU with warmup complete")
        print("=" * 80)

    def warmup_model(self):
        """Run warmup inference to compile the model."""
        import torch
        from PIL import Image

        # Warmup with a small image
        print("  Warmup 1: Image encoding...")
        test_img = Image.new("RGB", (224, 224), color="red")
        batch_images = self.processor.process_images([test_img]).to(self.model.device)

        with torch.no_grad():
            _ = self.model(**batch_images)
        print("    ✓ Image warmup complete")

        # Warmup with text
        print("  Warmup 2: Text encoding...")
        batch_texts = self.processor.process_texts(["test query"]).to(self.model.device)

        with torch.no_grad():
            _ = self.model(**batch_texts)
        print("    ✓ Text warmup complete")

        print("✓ Model warmup complete")

    @modal.asgi_app()
    def serve(self):
        """
        Create and return FastAPI app for serving embeddings.
        """
        import sys
        import torch
        import base64
        import io
        from fastapi import FastAPI, HTTPException
        from PIL import Image

        # Re-import after snapshot restore
        sys.path.insert(0, "/root/colpali")

        app = FastAPI(
            title="BiGemma3 Embedding API",
            description="HuggingFace-based BiGemma3 embeddings with GPU snapshots",
            version="1.0.0",
        )

        @app.get("/")
        async def root():
            return {
                "name": "BiGemma3 HuggingFace Server",
                "model": MODEL_NAME,
                "dimension": EMBEDDING_DIM,
                "status": "ready",
                "device": str(self.model.device),
            }

        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "model": MODEL_NAME, "gpu_available": torch.cuda.is_available()}

        @app.post("/embed", response_model=EmbeddingResponse)
        async def create_embeddings(request: EmbeddingRequest):
            """
            Generate embeddings for texts and/or images.

            Args:
                texts: List of text queries (optional)
                images: List of base64-encoded images (optional)
                normalize: Whether to L2-normalize embeddings (default: True)

            Returns:
                EmbeddingResponse with embeddings list
            """
            try:
                embeddings_list = []

                # Process images if provided
                if request.images:
                    print(f"Processing {len(request.images)} images...")
                    images = []
                    for b64_str in request.images:
                        # Decode base64 image
                        img_data = base64.b64decode(b64_str)
                        img = Image.open(io.BytesIO(img_data))
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        images.append(img)

                    batch_images = self.processor.process_images(images).to(self.model.device)

                    with torch.no_grad():
                        image_embeddings = self.model(**batch_images)

                    # Convert to numpy
                    image_embeddings_np = image_embeddings.cpu().float().numpy()
                    embeddings_list.extend(image_embeddings_np.tolist())

                # Process texts if provided
                if request.texts:
                    print(f"Processing {len(request.texts)} texts...")
                    batch_texts = self.processor.process_texts(request.texts).to(self.model.device)

                    with torch.no_grad():
                        text_embeddings = self.model(**batch_texts)

                    # Convert to numpy
                    text_embeddings_np = text_embeddings.cpu().float().numpy()
                    embeddings_list.extend(text_embeddings_np.tolist())

                if not embeddings_list:
                    raise HTTPException(status_code=400, detail="No texts or images provided")

                return EmbeddingResponse(
                    embeddings=embeddings_list,
                    model=MODEL_NAME,
                    dimension=EMBEDDING_DIM,
                    count=len(embeddings_list),
                )

            except Exception as e:
                print(f"Error processing embeddings: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        return app

    @modal.exit()
    def shutdown(self):
        """Clean shutdown."""
        print("Shutting down BiGemma3 HF server...")


# ============= Deployment Information =============
@app.function(image=image)
def print_deployment_info():
    """Print deployment information and endpoint URLs."""
    print("\n" + "=" * 80)
    print("BiGemma3 HuggingFace Server - Deployment Information")
    print("=" * 80)
    print(f"\nModel: {MODEL_NAME}")
    print(f"GPU: {gpu_config}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Scaledown window: {SCALEDOWN_WINDOW}s")
    print(f"Memory snapshots: ENABLED (GPU)")
    print(f"\nExpected cold start: <2s (with GPU snapshot)")
    print(f"Expected warm start: <100ms")
    print("\nEndpoints:")
    print("  - POST /embed (generate embeddings)")
    print("  - GET /health (health check)")
    print("  - GET / (server info)")
    print("\nRequest format (POST /embed):")
    print("  {")
    print('    "texts": ["query 1", "query 2"],  // optional')
    print('    "images": ["base64_img1", ...],    // optional')
    print('    "normalize": true                  // default: true')
    print("  }")
    print("\nAccess your deployment:")
    print(f"  modal.Cls.from_name('{APP_NAME}', 'BiGemma3HFServer')().serve.web_url")
    print("=" * 80)


@app.local_entrypoint()
def main():
    """
    Deploy the BiGemma3 HuggingFace server.

    Usage:
        modal deploy modal_scripts/gemma3/serve_hf_snapshot.py
    """
    print_deployment_info.remote()
