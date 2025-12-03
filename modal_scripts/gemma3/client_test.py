"""
Client Test Script for BiGemma3 Serving Cold Start Benchmarking

This script tests cold start performance for both vLLM and HuggingFace deployments.
It measures:
- First request time (cold start)
- Subsequent request times (warm start)
- Time after idle period (scaledown -> cold start)

Usage:
    # Test vLLM endpoint
    python client_test.py --mode vllm

    # Test HuggingFace endpoint
    python client_test.py --mode huggingface

    # Test both and compare
    python client_test.py --mode both

    # Custom endpoint
    python client_test.py --mode custom --url https://your-endpoint.modal.run

    # Wait for scaledown and test cold start
    python client_test.py --mode vllm --test-scaledown --scaledown-wait 65

Results are saved to:
    - cold_start_results_vllm.json
    - cold_start_results_hf.json
"""

import argparse
import base64
import io
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import requests
from PIL import Image


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    request_num: int
    start_time: float
    end_time: float
    duration_ms: float
    status_code: int
    success: bool
    error: Optional[str] = None
    response_size: Optional[int] = None


@dataclass
class TestResults:
    """Complete test results for a deployment."""

    deployment_name: str
    endpoint_url: str
    test_timestamp: str
    num_requests: int
    cold_start_ms: float
    warm_start_avg_ms: float
    warm_start_min_ms: float
    warm_start_max_ms: float
    warm_start_p50_ms: float
    warm_start_p95_ms: float
    warm_start_p99_ms: float
    post_scaledown_cold_start_ms: Optional[float]
    all_requests: List[Dict]
    summary: str


def create_test_image(size=(224, 224), color="blue", text="Test"):
    """Create a synthetic test image."""
    img = Image.new("RGB", size, color=color)
    return img


def encode_image_b64(img: Image.Image) -> str:
    """Encode PIL image to base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def test_vllm_endpoint(
    url: str,
    num_warmup_requests: int = 10,
    test_scaledown: bool = False,
    scaledown_wait: int = 65,
) -> TestResults:
    """
    Test vLLM endpoint (OpenAI-compatible).

    Args:
        url: Endpoint URL
        num_warmup_requests: Number of warm requests to test
        test_scaledown: Whether to test cold start after scaledown
        scaledown_wait: Seconds to wait for scaledown (default: 65s for 60s idle timeout)
    """
    print("=" * 80)
    print("Testing vLLM Endpoint")
    print("=" * 80)
    print(f"URL: {url}")
    print(f"Warmup requests: {num_warmup_requests}")
    print(f"Test scaledown: {test_scaledown}")

    # Create test data
    test_img = create_test_image(size=(224, 224), color="green", text="vLLM Test")
    b64_img = encode_image_b64(test_img)

    # Prepare request payload
    def make_request():
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

        start = time.time()
        try:
            resp = requests.post(f"{url}/v1/embeddings", json=payload, timeout=300)
            end = time.time()

            success = resp.status_code == 200
            error = None if success else resp.text
            response_size = len(resp.content) if success else None

            return RequestMetrics(
                request_num=0,
                start_time=start,
                end_time=end,
                duration_ms=(end - start) * 1000,
                status_code=resp.status_code,
                success=success,
                error=error,
                response_size=response_size,
            )
        except Exception as e:
            end = time.time()
            return RequestMetrics(
                request_num=0,
                start_time=start,
                end_time=end,
                duration_ms=(end - start) * 1000,
                status_code=0,
                success=False,
                error=str(e),
            )

    # Test 1: Cold start (first request)
    print("\n--- Test 1: Cold Start (First Request) ---")
    print("Sending first request...")
    cold_start_metrics = make_request()
    cold_start_metrics.request_num = 1

    print(f"✓ Cold start time: {cold_start_metrics.duration_ms:.2f}ms")
    print(f"  Status: {cold_start_metrics.status_code}")
    print(f"  Success: {cold_start_metrics.success}")

    # Test 2: Warm requests
    print(f"\n--- Test 2: Warm Requests ({num_warmup_requests} requests) ---")
    warm_metrics = []
    for i in range(num_warmup_requests):
        print(f"Sending request {i+1}/{num_warmup_requests}...", end=" ")
        metrics = make_request()
        metrics.request_num = i + 2  # Start from 2 since cold start was 1
        warm_metrics.append(metrics)
        print(f"{metrics.duration_ms:.2f}ms")
        time.sleep(0.1)  # Small delay between requests

    # Calculate warm start statistics
    warm_durations = [m.duration_ms for m in warm_metrics if m.success]
    warm_durations_sorted = sorted(warm_durations)

    warm_avg = sum(warm_durations) / len(warm_durations) if warm_durations else 0
    warm_min = min(warm_durations) if warm_durations else 0
    warm_max = max(warm_durations) if warm_durations else 0
    warm_p50 = warm_durations_sorted[len(warm_durations_sorted) // 2] if warm_durations else 0
    warm_p95 = warm_durations_sorted[int(len(warm_durations_sorted) * 0.95)] if warm_durations else 0
    warm_p99 = warm_durations_sorted[int(len(warm_durations_sorted) * 0.99)] if warm_durations else 0

    print(f"\n✓ Warm start statistics:")
    print(f"  Average: {warm_avg:.2f}ms")
    print(f"  Min: {warm_min:.2f}ms")
    print(f"  Max: {warm_max:.2f}ms")
    print(f"  P50: {warm_p50:.2f}ms")
    print(f"  P95: {warm_p95:.2f}ms")
    print(f"  P99: {warm_p99:.2f}ms")

    # Test 3: Post-scaledown cold start (optional)
    post_scaledown_metrics = None
    if test_scaledown:
        print(f"\n--- Test 3: Post-Scaledown Cold Start ---")
        print(f"Waiting {scaledown_wait}s for server to scale down...")
        time.sleep(scaledown_wait)

        print("Sending request after scaledown...")
        post_scaledown_metrics = make_request()
        post_scaledown_metrics.request_num = len(warm_metrics) + 2

        print(f"✓ Post-scaledown cold start: {post_scaledown_metrics.duration_ms:.2f}ms")
        print(f"  Status: {post_scaledown_metrics.status_code}")
        print(f"  Success: {post_scaledown_metrics.success}")

    # Compile results
    all_requests = [cold_start_metrics] + warm_metrics
    if post_scaledown_metrics:
        all_requests.append(post_scaledown_metrics)

    summary = f"""
vLLM Endpoint Test Results:
- Cold start: {cold_start_metrics.duration_ms:.2f}ms
- Warm avg: {warm_avg:.2f}ms (P50: {warm_p50:.2f}ms, P95: {warm_p95:.2f}ms)
- Post-scaledown cold start: {post_scaledown_metrics.duration_ms if post_scaledown_metrics else 'N/A'}ms
- Success rate: {sum(1 for m in all_requests if m.success)}/{len(all_requests)}
"""

    return TestResults(
        deployment_name="vLLM",
        endpoint_url=url,
        test_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        num_requests=len(all_requests),
        cold_start_ms=cold_start_metrics.duration_ms,
        warm_start_avg_ms=warm_avg,
        warm_start_min_ms=warm_min,
        warm_start_max_ms=warm_max,
        warm_start_p50_ms=warm_p50,
        warm_start_p95_ms=warm_p95,
        warm_start_p99_ms=warm_p99,
        post_scaledown_cold_start_ms=post_scaledown_metrics.duration_ms if post_scaledown_metrics else None,
        all_requests=[asdict(m) for m in all_requests],
        summary=summary,
    )


def test_hf_endpoint(
    url: str,
    num_warmup_requests: int = 10,
    test_scaledown: bool = False,
    scaledown_wait: int = 65,
) -> TestResults:
    """
    Test HuggingFace endpoint (FastAPI).

    Args:
        url: Endpoint URL
        num_warmup_requests: Number of warm requests to test
        test_scaledown: Whether to test cold start after scaledown
        scaledown_wait: Seconds to wait for scaledown
    """
    print("=" * 80)
    print("Testing HuggingFace Endpoint")
    print("=" * 80)
    print(f"URL: {url}")
    print(f"Warmup requests: {num_warmup_requests}")
    print(f"Test scaledown: {test_scaledown}")

    # Create test data
    test_img = create_test_image(size=(224, 224), color="red", text="HF Test")
    b64_img = encode_image_b64(test_img)

    # Prepare request payload
    def make_request():
        payload = {"images": [b64_img], "normalize": True}

        start = time.time()
        try:
            resp = requests.post(f"{url}/embed", json=payload, timeout=300)
            end = time.time()

            success = resp.status_code == 200
            error = None if success else resp.text
            response_size = len(resp.content) if success else None

            return RequestMetrics(
                request_num=0,
                start_time=start,
                end_time=end,
                duration_ms=(end - start) * 1000,
                status_code=resp.status_code,
                success=success,
                error=error,
                response_size=response_size,
            )
        except Exception as e:
            end = time.time()
            return RequestMetrics(
                request_num=0,
                start_time=start,
                end_time=end,
                duration_ms=(end - start) * 1000,
                status_code=0,
                success=False,
                error=str(e),
            )

    # Test 1: Cold start
    print("\n--- Test 1: Cold Start (First Request) ---")
    print("Sending first request...")
    cold_start_metrics = make_request()
    cold_start_metrics.request_num = 1

    print(f"✓ Cold start time: {cold_start_metrics.duration_ms:.2f}ms")
    print(f"  Status: {cold_start_metrics.status_code}")
    print(f"  Success: {cold_start_metrics.success}")

    # Test 2: Warm requests
    print(f"\n--- Test 2: Warm Requests ({num_warmup_requests} requests) ---")
    warm_metrics = []
    for i in range(num_warmup_requests):
        print(f"Sending request {i+1}/{num_warmup_requests}...", end=" ")
        metrics = make_request()
        metrics.request_num = i + 2
        warm_metrics.append(metrics)
        print(f"{metrics.duration_ms:.2f}ms")
        time.sleep(0.1)

    # Calculate statistics
    warm_durations = [m.duration_ms for m in warm_metrics if m.success]
    warm_durations_sorted = sorted(warm_durations)

    warm_avg = sum(warm_durations) / len(warm_durations) if warm_durations else 0
    warm_min = min(warm_durations) if warm_durations else 0
    warm_max = max(warm_durations) if warm_durations else 0
    warm_p50 = warm_durations_sorted[len(warm_durations_sorted) // 2] if warm_durations else 0
    warm_p95 = warm_durations_sorted[int(len(warm_durations_sorted) * 0.95)] if warm_durations else 0
    warm_p99 = warm_durations_sorted[int(len(warm_durations_sorted) * 0.99)] if warm_durations else 0

    print(f"\n✓ Warm start statistics:")
    print(f"  Average: {warm_avg:.2f}ms")
    print(f"  Min: {warm_min:.2f}ms")
    print(f"  Max: {warm_max:.2f}ms")
    print(f"  P50: {warm_p50:.2f}ms")
    print(f"  P95: {warm_p95:.2f}ms")
    print(f"  P99: {warm_p99:.2f}ms")

    # Test 3: Post-scaledown
    post_scaledown_metrics = None
    if test_scaledown:
        print(f"\n--- Test 3: Post-Scaledown Cold Start ---")
        print(f"Waiting {scaledown_wait}s for server to scale down...")
        time.sleep(scaledown_wait)

        print("Sending request after scaledown...")
        post_scaledown_metrics = make_request()
        post_scaledown_metrics.request_num = len(warm_metrics) + 2

        print(f"✓ Post-scaledown cold start: {post_scaledown_metrics.duration_ms:.2f}ms")
        print(f"  Status: {post_scaledown_metrics.status_code}")
        print(f"  Success: {post_scaledown_metrics.success}")

    # Compile results
    all_requests = [cold_start_metrics] + warm_metrics
    if post_scaledown_metrics:
        all_requests.append(post_scaledown_metrics)

    summary = f"""
HuggingFace Endpoint Test Results:
- Cold start: {cold_start_metrics.duration_ms:.2f}ms
- Warm avg: {warm_avg:.2f}ms (P50: {warm_p50:.2f}ms, P95: {warm_p95:.2f}ms)
- Post-scaledown cold start: {post_scaledown_metrics.duration_ms if post_scaledown_metrics else 'N/A'}ms
- Success rate: {sum(1 for m in all_requests if m.success)}/{len(all_requests)}
"""

    return TestResults(
        deployment_name="HuggingFace",
        endpoint_url=url,
        test_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        num_requests=len(all_requests),
        cold_start_ms=cold_start_metrics.duration_ms,
        warm_start_avg_ms=warm_avg,
        warm_start_min_ms=warm_min,
        warm_start_max_ms=warm_max,
        warm_start_p50_ms=warm_p50,
        warm_start_p95_ms=warm_p95,
        warm_start_p99_ms=warm_p99,
        post_scaledown_cold_start_ms=post_scaledown_metrics.duration_ms if post_scaledown_metrics else None,
        all_requests=[asdict(m) for m in all_requests],
        summary=summary,
    )


def save_results(results: TestResults, filename: str):
    """Save test results to JSON file."""
    with open(filename, "w") as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\n✓ Results saved to {filename}")


def print_comparison(vllm_results: Optional[TestResults], hf_results: Optional[TestResults]):
    """Print comparison between vLLM and HuggingFace results."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    if vllm_results and hf_results:
        print(f"\n{'Metric':<30} {'vLLM':<20} {'HuggingFace':<20} {'Winner'}")
        print("-" * 80)

        metrics = [
            ("Cold Start", vllm_results.cold_start_ms, hf_results.cold_start_ms, "lower"),
            ("Warm Avg", vllm_results.warm_start_avg_ms, hf_results.warm_start_avg_ms, "lower"),
            ("Warm P50", vllm_results.warm_start_p50_ms, hf_results.warm_start_p50_ms, "lower"),
            ("Warm P95", vllm_results.warm_start_p95_ms, hf_results.warm_start_p95_ms, "lower"),
        ]

        for metric_name, vllm_val, hf_val, compare in metrics:
            vllm_str = f"{vllm_val:.2f}ms"
            hf_str = f"{hf_val:.2f}ms"
            if compare == "lower":
                winner = "✓ vLLM" if vllm_val < hf_val else "✓ HF"
            print(f"{metric_name:<30} {vllm_str:<20} {hf_str:<20} {winner}")

        # Post-scaledown comparison
        if vllm_results.post_scaledown_cold_start_ms and hf_results.post_scaledown_cold_start_ms:
            print("\nPost-Scaledown Cold Start:")
            print(f"  vLLM: {vllm_results.post_scaledown_cold_start_ms:.2f}ms")
            print(f"  HuggingFace: {hf_results.post_scaledown_cold_start_ms:.2f}ms")
            winner = (
                "vLLM"
                if vllm_results.post_scaledown_cold_start_ms < hf_results.post_scaledown_cold_start_ms
                else "HuggingFace"
            )
            print(f"  Winner: ✓ {winner}")


def main():
    parser = argparse.ArgumentParser(description="Test BiGemma3 serving cold start performance")
    parser.add_argument(
        "--mode",
        choices=["vllm", "huggingface", "both", "custom"],
        required=True,
        help="Which deployment to test",
    )
    parser.add_argument("--url", type=str, help="Custom endpoint URL (for mode=custom)")
    parser.add_argument("--num-warmup", type=int, default=10, help="Number of warmup requests (default: 10)")
    parser.add_argument("--test-scaledown", action="store_true", help="Test cold start after scaledown")
    parser.add_argument("--scaledown-wait", type=int, default=65, help="Seconds to wait for scaledown (default: 65)")
    parser.add_argument("--vllm-url", type=str, help="vLLM endpoint URL")
    parser.add_argument("--hf-url", type=str, help="HuggingFace endpoint URL")

    args = parser.parse_args()

    vllm_results = None
    hf_results = None

    # Get URLs from Modal if not provided
    if args.mode in ["vllm", "both"] and not args.vllm_url:
        print("Getting vLLM URL from Modal deployment...")
        try:
            import modal

            server = modal.Cls.from_name("bigemma3-vllm-serve", "BiGemma3Server")()
            args.vllm_url = server.serve.get_web_url()
            print(f"✓ vLLM URL: {args.vllm_url}")
        except Exception as e:
            print(f"✗ Failed to get vLLM URL: {e}")
            print("Please deploy first: modal deploy serve_vllm_snapshot.py")
            return

    if args.mode in ["huggingface", "both"] and not args.hf_url:
        print("Getting HuggingFace URL from Modal deployment...")
        try:
            import modal

            server = modal.Cls.from_name("bigemma3-hf-serve", "BiGemma3HFServer")()
            args.hf_url = server.serve.web_url
            print(f"✓ HuggingFace URL: {args.hf_url}")
        except Exception as e:
            print(f"✗ Failed to get HuggingFace URL: {e}")
            print("Please deploy first: modal deploy serve_hf_snapshot.py")
            return

    # Run tests
    if args.mode in ["vllm", "both"]:
        vllm_results = test_vllm_endpoint(
            args.vllm_url, args.num_warmup, args.test_scaledown, args.scaledown_wait
        )
        save_results(vllm_results, "cold_start_results_vllm.json")
        print(vllm_results.summary)

    if args.mode in ["huggingface", "both"]:
        hf_results = test_hf_endpoint(args.hf_url, args.num_warmup, args.test_scaledown, args.scaledown_wait)
        save_results(hf_results, "cold_start_results_hf.json")
        print(hf_results.summary)

    if args.mode == "custom":
        if not args.url:
            print("Error: --url required for custom mode")
            return
        # Try vLLM format first
        try:
            results = test_vllm_endpoint(args.url, args.num_warmup, args.test_scaledown, args.scaledown_wait)
            save_results(results, "cold_start_results_custom.json")
            print(results.summary)
        except Exception as e:
            print(f"vLLM format failed: {e}")
            print("Trying HuggingFace format...")
            results = test_hf_endpoint(args.url, args.num_warmup, args.test_scaledown, args.scaledown_wait)
            save_results(results, "cold_start_results_custom.json")
            print(results.summary)

    # Print comparison if both were tested
    if args.mode == "both":
        print_comparison(vllm_results, hf_results)

    print("\n" + "=" * 80)
    print("✓ TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
