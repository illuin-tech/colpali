from typing import Generator, cast

import pytest
import torch
from PIL import Image

from colpali_engine.models import ColQwen3_5Processor


@pytest.fixture(scope="module")
def model_name() -> str:
    return "athrael-soju/colqwen3.5-4.5B-v3"


@pytest.fixture(scope="module")
def processor_from_pretrained(model_name: str) -> Generator[ColQwen3_5Processor, None, None]:
    yield cast(ColQwen3_5Processor, ColQwen3_5Processor.from_pretrained(model_name))


def test_load_processor_from_pretrained(processor_from_pretrained: ColQwen3_5Processor):
    assert isinstance(processor_from_pretrained, ColQwen3_5Processor)


def test_process_images(processor_from_pretrained: ColQwen3_5Processor):
    image_size = (64, 32)
    image = Image.new("RGB", image_size, color="black")
    images = [image]

    batch_feature = processor_from_pretrained.process_images(images)

    assert "pixel_values" in batch_feature
    assert isinstance(batch_feature["pixel_values"], torch.Tensor)
    assert batch_feature["pixel_values"].shape[0] == len(images)
    assert batch_feature["pixel_values"].shape[1] >= 1
    assert batch_feature["pixel_values"].shape[-1] > 0


def test_process_texts(processor_from_pretrained: ColQwen3_5Processor):
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    batch_encoding = processor_from_pretrained.process_texts(queries)

    assert "input_ids" in batch_encoding
    assert isinstance(batch_encoding["input_ids"], torch.Tensor)
    assert cast(torch.Tensor, batch_encoding["input_ids"]).shape[0] == len(queries)


def test_process_queries(processor_from_pretrained: ColQwen3_5Processor):
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    batch_encoding = processor_from_pretrained.process_queries(queries)

    assert "input_ids" in batch_encoding
    assert isinstance(batch_encoding["input_ids"], torch.Tensor)
    assert cast(torch.Tensor, batch_encoding["input_ids"]).shape[0] == len(queries)
