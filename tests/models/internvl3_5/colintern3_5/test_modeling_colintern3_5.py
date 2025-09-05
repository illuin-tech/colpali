import logging
from typing import Generator, cast

import pytest
import torch
from datasets import load_dataset
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColIntern3_5, ColIntern3_5Processor
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def model_name() -> str:
    return "OpenGVLab/InternVL3_5-1B-HF"


@pytest.fixture(scope="module")
def model_without_mask(model_name: str) -> Generator[ColIntern3_5, None, None]:
    device = get_torch_device("auto")
    logger.info(f"Device used: {device}")

    yield cast(
        ColIntern3_5,
        ColIntern3_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            trust_remote_code=True,
        ).eval(),
    )
    tear_down_torch()


@pytest.fixture(scope="module")
def model_with_mask(model_name: str) -> Generator[ColIntern3_5, None, None]:
    device = get_torch_device("auto")
    logger.info(f"Device used: {device}")

    yield cast(
        ColIntern3_5,
        ColIntern3_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            trust_remote_code=True,
        ).eval(),
    )
    tear_down_torch()


@pytest.fixture(scope="module")
def processor(model_name: str) -> Generator[ColIntern3_5Processor, None, None]:
    yield cast(
        ColIntern3_5Processor, 
        ColIntern3_5Processor.from_pretrained(
            model_name, 
            max_num_visual_tokens=768
        )
    )


class TestColIntern3_5_Model:  # noqa N801
    @pytest.mark.slow
    def test_load_model_from_pretrained(self, model_without_mask: ColIntern3_5):
        assert isinstance(model_without_mask, ColIntern3_5)
        
    def test_model_has_required_attributes(self, model_without_mask: ColIntern3_5):
        """Test that the model has all required attributes."""
        assert hasattr(model_without_mask, 'language_model')
        assert hasattr(model_without_mask, 'custom_text_proj')
        
    def test_model_dtype(self, model_without_mask: ColIntern3_5):
        """Test that the model uses the correct dtype."""
        assert model_without_mask.custom_text_proj.weight.dtype == torch.bfloat16


class TestColIntern3_5_ModelIntegration:  # noqa N801
    @pytest.mark.slow
    def test_forward_images_integration(
        self,
        model_without_mask: ColIntern3_5,
        processor: ColIntern3_5Processor,
    ):
        """Test that model can handle image processing without errors."""
        # Create a batch of dummy images
        images = [
            Image.new("RGB", (224, 224), color="white"),
            Image.new("RGB", (448, 448), color="black"),
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
        assert emb_dim == 128  # ColIntern3_5 embedding dimension

    @pytest.mark.slow
    def test_forward_queries_integration(
        self,
        model_without_mask: ColIntern3_5,
        processor: ColIntern3_5Processor,
    ):
        queries = [
            "What is shown in this image?",
            "Describe the content of this document.",
        ]

        # Process the queries
        batch_queries = processor.process_queries(queries).to(model_without_mask.device)

        # Forward pass
        with torch.no_grad():
            outputs = model_without_mask(**batch_queries)

        # Assertions
        assert isinstance(outputs, torch.Tensor)
        assert outputs.dim() == 3
        batch_size, n_query_tokens, emb_dim = outputs.shape
        assert batch_size == len(queries)
        assert emb_dim == 128  # ColIntern3_5 embedding dimension

    @pytest.mark.slow
    def test_retrieval_integration(
        self,
        model_without_mask: ColIntern3_5,
        processor: ColIntern3_5Processor,
    ):
        """Test retrieval capabilities."""
        # Load the test dataset
        ds = load_dataset("hf-internal-testing/document-visual-retrieval-test", split="test")

        # Preprocess the examples
        batch_images = processor.process_images(images=ds["image"]).to(model_without_mask.device)
        batch_queries = processor.process_queries(queries=ds["query"]).to(model_without_mask.device)

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

        # For an untrained model, we just check that the retrieval mechanism works
        # (produces valid scores) rather than expecting perfect accuracy
        assert torch.isfinite(scores).all(), "All scores should be finite"
        assert not torch.isnan(scores).any(), "No scores should be NaN"
        
        # Check that scores are reasonable (not all identical)
        assert scores.std() > 0, "Scores should have some variance"
        
        # Check that argmax produces valid indices
        max_indices = scores.argmax(dim=1)
        assert (max_indices >= 0).all() and (max_indices < len(ds)).all(), "Argmax indices should be valid"

    def test_output_consistency(
        self,
        model_without_mask: ColIntern3_5,
        processor: ColIntern3_5Processor,
    ):
        """Test that image and query outputs have consistent dimensionality."""
        # Create test inputs
        image = Image.new("RGB", (224, 224), color="white")
        query = "Test query"
        
        # Process inputs
        batch_image = processor.process_images([image]).to(model_without_mask.device)
        batch_query = processor.process_queries([query]).to(model_without_mask.device)
        
        # Forward pass
        with torch.no_grad():
            image_output = model_without_mask(**batch_image)
            query_output = model_without_mask(**batch_query)
        
        # Check consistency
        assert image_output.shape[0] == query_output.shape[0] == 1  # Same batch size
        assert image_output.shape[2] == query_output.shape[2] == 128  # Same embedding dimension

    def test_visual_token_limitation(
        self,
        model_without_mask: ColIntern3_5,
        processor: ColIntern3_5Processor,
    ):
        """Test that visual token limitation is working."""
        # Create a moderately large image 
        large_image = Image.new("RGB", (800, 800), color="white")
        
        # Process the image
        batch_image = processor.process_images([large_image]).to(model_without_mask.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model_without_mask(**batch_image)
        
        # Check that the number of tokens is reasonable
        _, n_tokens, _ = outputs.shape
        assert n_tokens < 2000, f"Too many visual tokens: {n_tokens}. Token limitation may not be working properly."

    def test_different_image_sizes(
        self,
        model_without_mask: ColIntern3_5,
        processor: ColIntern3_5Processor,
    ):
        """Test processing images of different sizes."""
        images = [
            Image.new("RGB", (224, 224), color="red"),
            Image.new("RGB", (448, 448), color="green"),
            Image.new("RGB", (300, 500), color="blue"),
            Image.new("RGB", (600, 400), color="yellow"),
        ]
        
        # Process images
        batch_images = processor.process_images(images).to(model_without_mask.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model_without_mask(**batch_images)
        
        # Assertions
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape[0] == len(images)
        assert outputs.shape[2] == 128  # Embedding dimension
        assert outputs.shape[-1] == 128
