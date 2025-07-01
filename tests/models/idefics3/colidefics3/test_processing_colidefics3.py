from typing import Generator, cast

import pytest
import torch
from PIL import Image

from colpali_engine.models import ColIdefics3Processor


@pytest.fixture(scope="module")
def model_name() -> str:
    return "vidore/colSmol-256M"


@pytest.fixture(scope="module")
def processor_from_pretrained(model_name: str) -> Generator[ColIdefics3Processor, None, None]:
    yield cast(ColIdefics3Processor, ColIdefics3Processor.from_pretrained(model_name))


def test_load_processor_from_pretrained(processor_from_pretrained: ColIdefics3Processor):
    assert isinstance(processor_from_pretrained, ColIdefics3Processor)


def test_process_images(processor_from_pretrained: ColIdefics3Processor):
    # Create a dummy image
    image_size = (16, 32)
    image = Image.new("RGB", image_size, color="black")
    images = [image]

    # Process the image
    batch_feature = processor_from_pretrained.process_images(images)

    # Assertions
    assert "pixel_values" in batch_feature
    assert batch_feature["pixel_values"].shape == torch.Size([1, 9, 3, 512, 512])


def test_process_texts(processor_from_pretrained: ColIdefics3Processor):
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the texts
    batch_encoding = processor_from_pretrained.process_texts(queries)

    # Assertions
    assert "input_ids" in batch_encoding
    assert isinstance(batch_encoding["input_ids"], torch.Tensor)
    assert cast(torch.Tensor, batch_encoding["input_ids"]).shape[0] == len(queries)


def test_process_queries(processor_from_pretrained: ColIdefics3Processor):
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
