import logging
from typing import Generator, cast

import pytest
import torch
from datasets import load_dataset
from PIL import Image

from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def model_name() -> str:
    return "ModernVBERT/colmodernvbert"


@pytest.fixture(scope="module")
def model_without_mask(model_name: str) -> Generator[ColModernVBert, None, None]:
    device = get_torch_device("auto")
    logger.info(f"Device used: {device}")

    yield cast(
        ColModernVBert,
        ColModernVBert.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
            attn_implementation="eager",
            mask_non_image_embeddings=False,
        ).eval(),
    )
    tear_down_torch()


@pytest.fixture(scope="module")
def model_with_mask(model_name: str) -> Generator[ColModernVBert, None, None]:
    device = get_torch_device("auto")
    logger.info(f"Device used: {device}")

    yield cast(
        ColModernVBert,
        ColModernVBert.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
            attn_implementation="eager",
            mask_non_image_embeddings=True,
        ).eval(),
    )
    tear_down_torch()


@pytest.fixture(scope="module")
def processor(model_name: str) -> Generator[ColModernVBertProcessor, None, None]:
    yield cast(ColModernVBertProcessor, ColModernVBertProcessor.from_pretrained(model_name))


class TestColModernVBert_Model:  # noqa N801
    @pytest.mark.slow
    def test_load_model_from_pretrained(self, model_without_mask: ColModernVBert):
        assert isinstance(model_without_mask, ColModernVBert)


class TestColModernVBert_ModelIntegration:  # noqa N801
    @pytest.mark.slow
    def test_forward_images_integration(
        self,
        model_without_mask: ColModernVBert,
        processor: ColModernVBertProcessor,
    ):
        # Create a batch of dummy images
        images = [
            Image.new("RGB", (64, 64), color="white"),
            Image.new("RGB", (32, 32), color="black"),
        ]

        # Process the image
        batch_images = processor.process_images(images).to(model_without_mask.device)

        # Forward pass
        with torch.no_grad():
            outputs = model_without_mask(**batch_images)

        # Assertions
        assert isinstance(outputs, torch.Tensor)
        assert outputs.dim() == 3
        batch_size, n_visual_tokens, emb_dim = outputs.shape
        assert batch_size == len(images)
        assert emb_dim == model_without_mask.dim

    @pytest.mark.slow
    def test_forward_queries_integration(
        self,
        model_without_mask: ColModernVBert,
        processor: ColModernVBertProcessor,
    ):
        queries = [
            "Is attention really all you need?",
            "Are Benjamin, Antoine, Merve, and Jo best friends?",
        ]

        # Process the queries
        batch_queries = processor.process_queries(queries).to(model_without_mask.device).to(torch.float32)

        # Forward pass
        with torch.no_grad():
            outputs = model_without_mask(**batch_queries)

        # Assertions
        assert isinstance(outputs, torch.Tensor)
        assert outputs.dim() == 3
        batch_size, n_query_tokens, emb_dim = outputs.shape
        assert batch_size == len(queries)
        assert emb_dim == model_without_mask.dim

    @pytest.mark.slow
    def test_retrieval_integration(
        self,
        model_without_mask: ColModernVBert,
        processor: ColModernVBertProcessor,
    ):
        # Load the test dataset
        ds = load_dataset("hf-internal-testing/document-visual-retrieval-test", split="test")

        # Preprocess the examples
        batch_images = processor.process_images(images=ds["image"]).to(model_without_mask.device).to(torch.float32)
        batch_queries = processor.process_queries(queries=ds["query"]).to(model_without_mask.device).to(torch.float32)

        # Run inference
        with torch.inference_mode():
            image_embeddings = model_without_mask(**batch_images)
            query_embeddings = model_without_mask(**batch_queries)

        # Compute retrieval scores
        scores = processor.score_multi_vector(
            qs=query_embeddings,
            ps=image_embeddings,
        )  # (len(qs), len(ps))

        assert scores.ndim == 2, f"Expected 2D tensor, got {scores.ndim}"
        assert scores.shape == (len(ds), len(ds)), f"Expected shape {(len(ds), len(ds))}, got {scores.shape}"
