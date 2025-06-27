import logging
from typing import Generator, cast

import pytest
import torch
from datasets import load_dataset
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def model_name() -> str:
    return "vidore/colqwen2-v1.0"


@pytest.fixture(scope="module")
def model(model_name: str) -> Generator[ColQwen2, None, None]:
    device = get_torch_device("auto")
    logger.info(f"Device used: {device}")

    yield cast(
        ColQwen2,
        ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval(),
    )
    tear_down_torch()


@pytest.fixture(scope="module")
def processor(model_name: str) -> Generator[ColQwen2Processor, None, None]:
    yield cast(ColQwen2Processor, ColQwen2Processor.from_pretrained(model_name))


class TestColQwen2Model:
    @pytest.mark.slow
    def test_load_model_from_pretrained(self, model: ColQwen2):
        assert isinstance(model, ColQwen2)


class TestColQwen2ModelIntegration:
    @pytest.mark.slow
    def test_forward_images_integration(
        self,
        model: ColQwen2,
        processor: ColQwen2Processor,
    ):
        # Create a batch of dummy images
        images = [
            Image.new("RGB", (64, 64), color="white"),
            Image.new("RGB", (32, 32), color="black"),
        ]

        # Process the image
        batch_images = processor.process_images(images).to(model.device)

        # Forward pass
        model.mask_non_image_embeddings = False
        with torch.no_grad():
            outputs = model(**batch_images)

        # Assertions
        assert isinstance(outputs, torch.Tensor)
        assert outputs.dim() == 3
        batch_size, n_visual_tokens, emb_dim = outputs.shape
        assert batch_size == len(images)
        assert emb_dim == model.dim

    @pytest.mark.slow
    def test_forward_queries_integration(
        self,
        model: ColQwen2,
        processor: ColQwen2Processor,
    ):
        queries = [
            "Is attention really all you need?",
            "Are Benjamin, Antoine, Merve, and Jo best friends?",
        ]

        # Process the queries
        batch_queries = processor.process_queries(queries).to(model.device)

        # Forward pass
        with torch.no_grad():
            outputs = model(**batch_queries)

        # Assertions
        assert isinstance(outputs, torch.Tensor)
        assert outputs.dim() == 3
        batch_size, n_query_tokens, emb_dim = outputs.shape
        assert batch_size == len(queries)
        assert emb_dim == model.dim

    @pytest.mark.slow
    def test_retrieval_integration(
        self,
        model: ColQwen2,
        processor: ColQwen2Processor,
    ):
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
