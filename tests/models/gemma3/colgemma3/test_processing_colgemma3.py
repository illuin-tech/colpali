from typing import Generator, cast

import pytest
import torch
from PIL import Image

from colpali_engine.models import ColGemmaProcessor3


@pytest.fixture(scope="module")
def model_name() -> str:
    return "Cognitive-Lab/ColNetraEmbed"


@pytest.fixture(scope="module")
def processor_from_pretrained(model_name: str) -> Generator[ColGemmaProcessor3, None, None]:
    yield cast(ColGemmaProcessor3, ColGemmaProcessor3.from_pretrained(model_name, use_fast=True))


def test_load_processor_from_pretrained(processor_from_pretrained: ColGemmaProcessor3):
    assert isinstance(processor_from_pretrained, ColGemmaProcessor3)


def test_process_images(processor_from_pretrained: ColGemmaProcessor3):
    # Create a dummy image
    image_size = (16, 32)
    image = Image.new("RGB", image_size, color="black")
    images = [image]

    # Process the image
    batch_feature = processor_from_pretrained.process_images(images)

    # Assertions
    assert "pixel_values" in batch_feature
    assert isinstance(batch_feature["pixel_values"], torch.Tensor)
    assert batch_feature["pixel_values"].shape[0] == 1
    assert "input_ids" in batch_feature
    assert isinstance(batch_feature["input_ids"], torch.Tensor)
    assert batch_feature["input_ids"].shape[0] == 1


def test_process_texts(processor_from_pretrained: ColGemmaProcessor3):
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


def test_process_queries(processor_from_pretrained: ColGemmaProcessor3):
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
