import logging
from typing import Generator, List, cast

import pytest
import torch
from PIL import Image
from transformers import BatchFeature
from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig

from colpali_engine.models import ColPali2, ColPali2Config, ColPali2Processor
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def colpali_2_config() -> Generator[ColPali2Config, None, None]:
    yield ColPali2Config(
        vlm_backbone_config=cast(
            PaliGemmaConfig,
            PaliGemmaConfig.from_pretrained("google/paligemma-3b-mix-448"),
        ),
        single_vector_projector_dim=128,
        single_vector_pool_strategy="mean",
        multi_vector_projector_dim=128,
    )


@pytest.fixture(scope="module")
def colpali_2_from_config(colpali_2_config: ColPali2Config) -> Generator[ColPali2, None, None]:
    device = get_torch_device("auto")
    logger.info(f"Device used: {device}")

    yield ColPali2(config=colpali_2_config)
    tear_down_torch()


@pytest.fixture(scope="module")
def colpali_2_model_path() -> str:
    raise NotImplementedError("Please provide the path to the model in the hub")


@pytest.fixture(scope="module")
def colpali_2_from_pretrained(colpali_2_model_path: str) -> Generator[ColPali2, None, None]:
    device = get_torch_device("auto")
    logger.info(f"Device used: {device}")

    yield cast(
        ColPali2,
        ColPali2.from_pretrained(
            colpali_2_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ),
    )
    tear_down_torch()


@pytest.fixture(scope="class")
def processor() -> Generator[ColPali2Processor, None, None]:
    yield cast(ColPali2Processor, ColPali2Processor.from_pretrained("google/paligemma-3b-mix-448"))


@pytest.fixture(scope="class")
def images() -> Generator[List[Image.Image], None, None]:
    yield [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (16, 16), color="black"),
    ]


@pytest.fixture(scope="class")
def queries() -> Generator[List[str], None, None]:
    yield [
        "Does Manu like to play football?",
        "Are Benjamin, Antoine, Merve, and Jo friends?",
        "Is byaldi a dish or an awesome repository for RAG?",
    ]


@pytest.fixture(scope="function")
def batch_queries(processor: ColPali2Processor, queries: List[str]) -> Generator[BatchFeature, None, None]:
    yield processor.process_queries(queries)


@pytest.fixture(scope="function")
def batch_images(processor: ColPali2Processor, images: List[Image.Image]) -> Generator[BatchFeature, None, None]:
    yield processor.process_images(images)


class TestLoadColPali2:
    """
    Test the different ways to load ColPali2.
    """

    @pytest.mark.slow
    def test_load_colpali_2_from_config(self, colpali_2_config: ColPali2Config):
        device = get_torch_device("auto")
        logger.info(f"Device used: {device}")

        model = ColPali2(config=colpali_2_config)

        assert isinstance(model, ColPali2)
        assert model.single_vector_projector_dim == colpali_2_config.single_vector_projector_dim
        assert model.multi_vector_pooler.pooling_strategy == colpali_2_config.single_vector_pool_strategy
        assert model.multi_vector_projector_dim == colpali_2_config.multi_vector_projector_dim

        tear_down_torch()

    @pytest.mark.slow
    def test_load_colpali_2_from_pretrained(self, colpali_2_from_config: ColPali2):
        assert isinstance(colpali_2_from_config, ColPali2)


class TestForwardSingleVector:
    """
    Test the forward pass of ColPali2 for single-vector embeddings.
    """

    @pytest.mark.slow
    def test_colpali_2_forward_images(
        self,
        colpali_2_from_config: ColPali2,
        batch_images: BatchFeature,
    ):
        # Forward pass
        with torch.no_grad():
            outputs = colpali_2_from_config.forward_single_vector(**batch_images)

        # Assertions
        assert isinstance(outputs.single_vec_emb, torch.Tensor)
        assert outputs.single_vec_emb.dim() == 2
        batch_size, emb_dim = outputs.single_vec_emb.shape
        assert batch_size == len(batch_images["input_ids"])
        assert emb_dim == colpali_2_from_config.single_vector_projector_dim

    @pytest.mark.slow
    def test_colpali_2_forward_queries(
        self,
        colpali_2_from_config: ColPali2,
        batch_queries: BatchFeature,
    ):
        # Forward pass
        with torch.no_grad():
            outputs = colpali_2_from_config.forward_single_vector(**batch_queries)

        # Assertions
        assert isinstance(outputs.single_vec_emb, torch.Tensor)
        assert outputs.single_vec_emb.dim() == 2
        batch_size, emb_dim = outputs.single_vec_emb.shape
        assert batch_size == len(batch_queries["input_ids"])
        assert emb_dim == colpali_2_from_config.single_vector_projector_dim


class TestForwardMultiVector:
    """
    Test the forward pass of ColPali2 for multi-vector embeddings.
    """

    @pytest.mark.slow
    def test_colpali_2_forward_images(
        self,
        colpali_2_from_config: ColPali2,
        batch_images: BatchFeature,
    ):
        # Forward pass
        with torch.no_grad():
            outputs = colpali_2_from_config.forward_multi_vector(**batch_images)

        # Assertions
        assert isinstance(outputs.multi_vec_emb, torch.Tensor)
        assert outputs.multi_vec_emb.dim() == 3
        batch_size, n_visual_tokens, emb_dim = outputs.multi_vec_emb.shape
        assert batch_size == len(batch_images["input_ids"])
        assert emb_dim == colpali_2_from_config.multi_vector_projector_dim

    @pytest.mark.slow
    def test_colpali_2_forward_queries(
        self,
        colpali_2_from_config: ColPali2,
        batch_queries: BatchFeature,
    ):
        # Forward pass
        with torch.no_grad():
            outputs = colpali_2_from_config.forward_multi_vector(**batch_queries)

        # Assertions
        assert isinstance(outputs.multi_vec_emb, torch.Tensor)
        assert outputs.multi_vec_emb.dim() == 3
        batch_size, n_query_tokens, emb_dim = outputs.multi_vec_emb.shape
        assert batch_size == len(batch_queries["input_ids"])
        assert emb_dim == colpali_2_from_config.multi_vector_projector_dim
