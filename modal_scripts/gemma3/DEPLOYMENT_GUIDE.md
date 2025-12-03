# BiGemma3 Serving with GPU Snapshots - Deployment Guide

This guide covers deploying and testing BiGemma3 embedding servers with GPU memory snapshots for sub-2-second cold starts.

## Overview

Two deployment options are available:
1. **vLLM Server**: OpenAI-compatible API, optimized throughput
2. **HuggingFace Server**: FastAPI-based, direct colpali_engine integration

Both use:
- **GPU Memory Snapshots** for <2s cold starts
- **1-minute scaledown window** to balance cost and performance
- **L40S GPU** for optimal performance/cost ratio
- **Pre-compiled models** captured in GPU snapshot

## Architecture

### vLLM Server (`serve_vllm_snapshot.py`)

```
@enter(snap=True):  Start server → Compile → Warmup → Sleep → Snapshot
@enter(snap=False): Wake from sleep on new request
@web_server:        Expose OpenAI-compatible /v1/embeddings
```

**Features:**
- OpenAI-compatible API format
- Sleep mode for GPU memory preservation
- Supports text and multimodal (image) embeddings
- Batch processing via vLLM engine

### HuggingFace Server (`serve_hf_snapshot.py`)

```
@enter(snap=True):  Load model → Move to GPU → Warmup → Snapshot
@asgi_app:          Expose FastAPI /embed endpoint
```

**Features:**
- Simple REST API with FastAPI
- Direct BiGemma3/BiGemmaProcessor3 usage
- Batched inference support
- Base64 image input

## Prerequisites

```bash
# Install Modal
pip install modal

# Configure Modal account
modal setup

# Configure secrets in Modal dashboard
modal secret create adithya-hf-wandb HUGGINGFACE_TOKEN=hf_xxx
```

## Deployment

### Option 1: vLLM Server

```bash
# Deploy the server (creates GPU snapshot on first deploy)
modal deploy colpali/modal_scripts/gemma3/serve_vllm_snapshot.py

# The first deployment will:
# 1. Load model weights
# 2. Start vLLM server
# 3. Trigger compilation and CUDA graph capture
# 4. Create GPU memory snapshot
#
# This takes ~5-10 minutes for first deployment
# Subsequent cold starts will be <2 seconds!
```

**Endpoints:**
- `POST /v1/embeddings` - Generate embeddings (OpenAI format)
- `GET /health` - Health check
- `POST /sleep?level=1` - Manual sleep
- `POST /wake_up` - Manual wake

### Option 2: HuggingFace Server

```bash
# Deploy the server (creates GPU snapshot on first deploy)
modal deploy colpali/modal_scripts/gemma3/serve_hf_snapshot.py

# First deployment process:
# 1. Load BiGemma3 model onto GPU
# 2. Run warmup inference
# 3. Create GPU memory snapshot
#
# Takes ~3-5 minutes, then <2s cold starts
```

**Endpoints:**
- `POST /embed` - Generate embeddings
- `GET /health` - Health check
- `GET /` - Server info

## Testing Cold Start Performance

### Quick Test

```bash
# Test vLLM endpoint
python colpali/modal_scripts/gemma3/client_test.py --mode vllm --num-warmup 10

# Test HuggingFace endpoint
python colpali/modal_scripts/gemma3/client_test.py --mode huggingface --num-warmup 10

# Test both and compare
python colpali/modal_scripts/gemma3/client_test.py --mode both --num-warmup 10
```

### Comprehensive Test (with Scaledown)

```bash
# Test cold start after 60s idle timeout
python colpali/modal_scripts/gemma3/client_test.py \
    --mode both \
    --num-warmup 10 \
    --test-scaledown \
    --scaledown-wait 65
```

This will:
1. Measure initial cold start (snapshot restore)
2. Run 10 warm requests
3. Wait 65 seconds for scaledown
4. Measure post-scaledown cold start

### Expected Results

**With GPU Snapshots:**
- Initial cold start: **1-2 seconds**
- Warm requests: **50-200ms**
- Post-scaledown cold start: **1-2 seconds**

**Without GPU Snapshots (baseline):**
- Cold start: **15-30 seconds**
- Warm requests: **50-200ms**

## API Usage Examples

### vLLM Endpoint (OpenAI-compatible)

```python
import requests
import base64
import io
from PIL import Image

# Encode image
img = Image.open("test.jpg")
buffered = io.BytesIO()
img.save(buffered, format="JPEG")
b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

# Request embeddings
url = "https://your-deployment.modal.run"
payload = {
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

response = requests.post(f"{url}/v1/embeddings", json=payload)
embeddings = response.json()["data"][0]["embedding"]
print(f"Embedding dimension: {len(embeddings)}")  # 2560
```

### HuggingFace Endpoint (FastAPI)

```python
import requests
import base64
import io
from PIL import Image

# Encode image
img = Image.open("test.jpg")
buffered = io.BytesIO()
img.save(buffered, format="JPEG")
b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

# Request embeddings
url = "https://your-deployment.modal.run"
payload = {
    "images": [b64_img],
    "texts": ["query text"],  # Optional
    "normalize": True
}

response = requests.post(f"{url}/embed", json=payload)
result = response.json()
print(f"Embeddings: {len(result['embeddings'])}")
print(f"Dimension: {result['dimension']}")  # 2560
```

## Performance Optimization

### Scaledown Window

The default is 60 seconds (1 minute idle timeout):

```python
# In serve_*.py
SCALEDOWN_WINDOW = 60  # seconds
```

**Tuning recommendations:**
- **High traffic**: Increase to 300s (5 min) to keep warm longer
- **Low traffic**: Keep at 60s (1 min) to minimize costs
- **Cost-sensitive**: Decrease to 30s for aggressive scaledown

### GPU Memory Utilization

**vLLM:**
```python
"--gpu-memory-utilization", "0.6"  # 60% of GPU memory
```

**HuggingFace:**
```python
# Model loaded in bfloat16
dtype=torch.bfloat16
```

**Tuning:**
- Increase if you see OOM errors
- Decrease to allow more concurrent instances

### Batch Size

**vLLM:**
```python
"--max-num-seqs", "8"  # Max concurrent sequences
```

**HuggingFace:**
- Processes batches via request body (images/texts lists)

## Monitoring

### Health Checks

```bash
# vLLM
curl https://your-vllm.modal.run/health

# HuggingFace
curl https://your-hf.modal.run/health
```

### Modal Dashboard

Monitor your deployments:
```bash
modal app list
modal container list
modal container logs <container-id>
```

### Cold Start Metrics

Results are saved to:
- `cold_start_results_vllm.json`
- `cold_start_results_hf.json`

Example output:
```json
{
  "deployment_name": "vLLM",
  "cold_start_ms": 1853.2,
  "warm_start_avg_ms": 127.4,
  "warm_start_p50_ms": 118.3,
  "warm_start_p95_ms": 245.8,
  "post_scaledown_cold_start_ms": 1921.5,
  "summary": "..."
}
```

## Troubleshooting

### Issue: Cold start still slow (>5s)

**Cause:** GPU snapshot not created or invalidated

**Solution:**
1. Redeploy to create new snapshot:
   ```bash
   modal deploy serve_vllm_snapshot.py
   ```
2. Check logs for snapshot creation:
   ```bash
   modal app logs bigemma3-vllm-serve
   ```

### Issue: "CUDA not available" error

**Cause:** torch.cuda initialized during snapshot phase

**Solution:** Already handled via `TORCHINDUCTOR_COMPILE_THREADS=1` env var

### Issue: Request timeout

**Cause:** First request may take longer (compilation)

**Solution:** Increase timeout to 300s for first request:
```python
response = requests.post(url, json=payload, timeout=300)
```

### Issue: 500 Internal Server Error

**Cause:** Model failed to load or process input

**Solution:**
1. Check container logs:
   ```bash
   modal container list
   modal container logs <container-id>
   ```
2. Verify HuggingFace token is set correctly
3. Ensure image format is correct (RGB, JPEG)

## Cost Optimization

### Estimated Costs (Modal pricing)

**L40S GPU:** ~$1.10/hour

**Scenarios:**

1. **Always-on (no scaledown):**
   - 24/7 uptime: ~$792/month
   - Use case: High-traffic production

2. **60s scaledown (default):**
   - 10 requests/day, 5s each: ~$0.50/month
   - 100 requests/day, 5s each: ~$5/month
   - 1000 requests/day, 5s each: ~$50/month
   - Use case: Moderate traffic

3. **Aggressive scaledown (30s):**
   - Minimal idle time
   - Use case: Development, testing, low traffic

### Best Practices

1. **Development:** Use 30s scaledown
2. **Staging:** Use 60s scaledown
3. **Production (bursty):** Use 60-120s scaledown
4. **Production (steady):** Use 300s scaledown or min_containers=1

## References

- [Modal GPU Snapshots Documentation](https://modal.com/docs/guide/memory-snapshot)
- [GPU Memory Snapshots Blog Post](https://modal.com/blog/gpu-mem-snapshots)
- [Cold Start Performance Guide](https://modal.com/docs/guide/cold-start)
- [vLLM Documentation](https://docs.vllm.ai/)
- [BiGemma3 Model Card](https://huggingface.co/Cognitive-Lab/NetraEmbed)

## Support

For issues or questions:
- Modal: https://modal.com/docs
- BiGemma3/ColPali: https://github.com/illuin-tech/colpali

---

**Last Updated:** December 2024
**Modal Version:** 0.64+
**vLLM Version:** 0.11.0+
