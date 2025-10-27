from typing import Generator, cast

import pytest
import torch
from PIL import Image

from colpali_engine.models import ColModernVBertProcessor


@pytest.fixture(scope="module")
def model_name() -> str:
    return "ModernVBERT/colmodernvbert"


@pytest.fixture(scope="module")
def processor_from_pretrained(model_name: str) -> Generator[ColModernVBertProcessor, None, None]:
    yield cast(ColModernVBertProcessor, ColModernVBertProcessor.from_pretrained(model_name))


def test_load_processor_from_pretrained(processor_from_pretrained: ColModernVBertProcessor):
    assert isinstance(processor_from_pretrained, ColModernVBertProcessor)


def test_process_images(processor_from_pretrained: ColModernVBertProcessor):
    # Create a dummy image
    image_size = (64, 32)
    image = Image.new("RGB", image_size, color="black")
    images = [image]

    # Process the image
    batch_feature = processor_from_pretrained.process_images(images)

    # Assertions
    assert "pixel_values" in batch_feature
    assert isinstance(batch_feature["pixel_values"], torch.Tensor)
    assert batch_feature["pixel_values"].shape[0] == 1


def test_process_texts(processor_from_pretrained: ColModernVBertProcessor):
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the queries
    batch_encoding = processor_from_pretrained.process_texts(queries)

    # Assertions
    assert "input_ids" in batch_encoding
    assert isinstance(batch_encoding["input_ids"], torch.Tensor)
    assert cast(torch.Tensor, batch_encoding["input_ids"]).shape[0] == len(queries)


def test_process_queries(processor_from_pretrained: ColModernVBertProcessor):
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the queries
    batch_encoding = processor_from_pretrained.process_queries(queries)

    # Assertions
    assert "input_ids" in batch_encoding
    assert isinstance(batch_encoding["input_ids"], torch.Tensor)
    assert cast(torch.Tensor, batch_encoding["input_ids"]).shape[0] == len(queries)
