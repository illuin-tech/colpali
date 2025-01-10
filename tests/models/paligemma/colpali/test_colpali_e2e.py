from typing import cast

import pytest
import torch
from datasets import load_dataset

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device


@pytest.fixture(scope="module")
def model_name() -> str:
    return "vidore/colpali-v1.2"


@pytest.mark.slow
def test_e2e_retrieval_and_scoring(model_name: str):
    # Load the model and processor
    model = cast(
        ColPali,
        ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=get_torch_device("auto"),
        ),
    ).eval()
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))

    # Load the test dataset
    ds = load_dataset("hf-internal-testing/document-visual-retrieval-test", split="test")

    # Preprocess the examples
    batch_images = processor.process_images(images=ds["image"]).to(model.device)
    batch_queries = processor.process_queries(queries=ds["query"]).to(model.device)

    # Run inference
    with torch.inference_mode():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    # Compute retrieval scores
    scores = processor.score_multi_vector(
        qs=query_embeddings,
        ps=image_embeddings,
    )  # (len(qs), len(ps))

    assert scores.ndim == 2, f"Expected 2D tensor, got {scores.ndim}"
    assert scores.shape == (len(ds), len(ds)), f"Expected shape {(len(ds), len(ds))}, got {scores.shape}"

    # Check if the maximum scores per row are in the diagonal of the matrix score
    assert (scores.argmax(dim=1) == torch.arange(len(ds), device=scores.device)).all()

    # Further validation: fine-grained check, with a hardcoded score from the original implementation
    expected_scores = torch.tensor(
        [
            [16.5000, 7.5938, 15.6875],
            [12.0625, 16.2500, 11.1250],
            [15.2500, 12.6250, 21.0000],
        ],
        dtype=scores.dtype,
    )
    assert torch.allclose(scores, expected_scores, atol=1), f"Expected scores {expected_scores}, got {scores}"
