import argparse
import statistics
from time import perf_counter

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from colpali_engine.models import ColModernVBert, ColModernVBertProcessor


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_forward(
    model: ColModernVBert,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    *,
    iterations: int,
    warmup: int,
) -> tuple[torch.Tensor, list[float]]:
    output = None

    with torch.inference_mode():
        for _ in range(warmup):
            output = model(**inputs)
        synchronize(device)

        timings = []
        for _ in range(iterations):
            start = perf_counter()
            output = model(**inputs)
            synchronize(device)
            timings.append(perf_counter() - start)

    if output is None:
        raise RuntimeError("Benchmark did not run; use at least one warmup or iteration.")
    return output, timings


def print_stats(label: str, timings: list[float]) -> None:
    timings_ms = [elapsed * 1_000 for elapsed in timings]
    print(
        f"{label}: "
        f"mean={statistics.mean(timings_ms):.2f} ms, "
        f"median={statistics.median(timings_ms):.2f} ms, "
        f"min={min(timings_ms):.2f} ms, "
        f"max={max(timings_ms):.2f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ColModernVBERT text and image embedding latency.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of timed forward passes per input type.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of untimed warmup forward passes per input type.")
    args = parser.parse_args()

    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    model_id = "ModernVBERT/colmodernvbert"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = ColModernVBertProcessor.from_pretrained(model_id)
    model = ColModernVBert.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    image = Image.open(
        hf_hub_download(
            "HuggingFaceTB/SmolVLM",
            "example_images/rococo.jpg",
            repo_type="space",
        )
    )
    text = "This is a text"

    text_inputs = processor.process_texts([text]).to(device)
    image_inputs = processor.process_images([image]).to(device)

    q_embeddings, text_timings = benchmark_forward(
        model,
        text_inputs,
        device,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    corpus_embeddings, image_timings = benchmark_forward(
        model,
        image_inputs,
        device,
        iterations=args.iterations,
        warmup=args.warmup,
    )

    scores = processor.score(q_embeddings, corpus_embeddings)

    print(f"Device: {device}")
    print(f"Iterations: {args.iterations} timed, {args.warmup} warmup")
    print(f"Query embeddings shape: {tuple(q_embeddings.shape)}")
    print(f"Image embeddings shape: {tuple(corpus_embeddings.shape)}")
    print("Similarity scores:", scores)
    print_stats("Text embedding", text_timings)
    print_stats("Image embedding", image_timings)


if __name__ == "__main__":
    main()
