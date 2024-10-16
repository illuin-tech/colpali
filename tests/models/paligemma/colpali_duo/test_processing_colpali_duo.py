from typing import Generator, cast

import pytest
import torch
from PIL import Image

from colpali_engine.models import ColPaliDuoProcessor


@pytest.fixture(scope="module")
def colpali_model_path() -> str:
    return "google/paligemma-3b-mix-448"


@pytest.fixture(scope="module")
def processor_from_pretrained(colpali_model_path: str) -> Generator[ColPaliDuoProcessor, None, None]:
    yield cast(ColPaliDuoProcessor, ColPaliDuoProcessor.from_pretrained(colpali_model_path))


def test_load_processor_from_pretrained(processor_from_pretrained: ColPaliDuoProcessor):
    assert isinstance(processor_from_pretrained, ColPaliDuoProcessor)


def test_process_images(processor_from_pretrained: ColPaliDuoProcessor):
    # Create a dummy image
    image = Image.new("RGB", (16, 16), color="black")
    images = [image]

    # Process the image
    batch_feature = processor_from_pretrained.process_images(images)

    # Assertions
    assert "pixel_values" in batch_feature
    assert batch_feature["pixel_values"].shape == torch.Size([1, 3, 448, 448])


def test_process_queries(processor_from_pretrained: ColPaliDuoProcessor):
    queries = [
        "Does Manu like to play football?",
        "Are Benjamin, Antoine, Merve, and Jo friends?",
        "Is byaldi a dish or a nice repository for RAG?",
    ]

    # Process the queries
    batch_encoding = processor_from_pretrained.process_queries(queries)

    # Assertions
    assert "input_ids" in batch_encoding
    assert isinstance(batch_encoding["input_ids"], torch.Tensor)
    assert cast(torch.Tensor, batch_encoding["input_ids"]).shape[0] == len(queries)
