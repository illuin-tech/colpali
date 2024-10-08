import logging
from typing import Generator, cast

import pytest
import torch
from PIL import Image

from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def model_name() -> str:
    return "vidore/colqwen2-v0.1"


@pytest.fixture(scope="module")
def model_from_pretrained(model_name: str) -> Generator[ColQwen2, None, None]:
    device = get_torch_device("auto")
    logger.info(f"Device used: {device}")

    yield cast(
        ColQwen2,
        ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval(),
    )
    tear_down_torch()


@pytest.fixture(scope="module")
def processor(model_name: str) -> Generator[ColQwen2Processor, None, None]:
    yield cast(ColQwen2Processor, ColQwen2Processor.from_pretrained(model_name))


@pytest.mark.slow
def test_load_from_pretrained(model_from_pretrained: ColQwen2):
    assert isinstance(model_from_pretrained, ColQwen2)


@pytest.mark.slow
def test_forward_images(
    model_from_pretrained: ColQwen2,
    processor: ColQwen2Processor,
):
    # Create a batch of dummy images
    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (16, 16), color="black"),
    ]

    # Process the image
    batch_images = processor.process_images(images).to(model_from_pretrained.device)

    # Forward pass
    with torch.no_grad():
        outputs = model_from_pretrained(**batch_images)

    # Assertions
    assert isinstance(outputs, torch.Tensor)
    assert outputs.dim() == 3
    batch_size, n_visual_tokens, emb_dim = outputs.shape
    assert batch_size == len(images)
    assert emb_dim == model_from_pretrained.dim


@pytest.mark.slow
def test_forward_queries(model_from_pretrained: ColQwen2, processor: ColQwen2Processor):
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the queries
    batch_queries = processor.process_queries(queries).to(model_from_pretrained.device)

    # Forward pass
    with torch.no_grad():
        outputs = model_from_pretrained(**batch_queries)

    # Assertions
    assert isinstance(outputs, torch.Tensor)
    assert outputs.dim() == 3
    batch_size, n_query_tokens, emb_dim = outputs.shape
    assert batch_size == len(queries)
    assert emb_dim == model_from_pretrained.dim


@pytest.mark.slow
def test_is_model_deterministic(model_from_pretrained: ColQwen2, processor: ColQwen2Processor):
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the queries
    batch_queries = processor.process_queries(queries).to(model_from_pretrained.device)

    # Forward pass
    with torch.no_grad():
        outputs = model_from_pretrained(**batch_queries)

    # Forward pass again
    with torch.no_grad():
        outputs_new = model_from_pretrained(**batch_queries)

    # Assertions
    assert torch.allclose(outputs, outputs_new, atol=1e-5)
