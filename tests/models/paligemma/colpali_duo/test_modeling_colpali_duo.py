import logging
from typing import Generator, List, cast

import pytest
import torch
from PIL import Image
from transformers import BatchFeature
from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig

from colpali_engine.models import ColPaliDuo, ColPaliDuoConfig, ColPaliDuoProcessor
from colpali_engine.models.paligemma.colpali_duo.modeling_colpali_duo import ColPaliDuoModelOutput
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def colpali_duo_config() -> Generator[ColPaliDuoConfig, None, None]:
    yield ColPaliDuoConfig(
        **PaliGemmaConfig.from_pretrained("google/paligemma-3b-mix-448").to_dict(),
        single_vector_projector_dim=128,
        single_vector_pool_strategy="mean",
        multi_vector_projector_dim=128,
    )


@pytest.fixture(scope="module")
def test_load_colpali_duo_from_pretrained(colpali_duo_config: ColPaliDuoConfig) -> Generator[ColPaliDuo, None, None]:
    device = get_torch_device("auto")
    logger.info(f"Device used: {device}")

    yield ColPaliDuo(config=colpali_duo_config)
    tear_down_torch()


@pytest.fixture(scope="module")
def colpali_duo_model_path() -> str:
    return "vidore/colpali-duo-base"


@pytest.fixture(scope="module")
def colpali_duo_from_pretrained(colpali_duo_model_path: str) -> Generator[ColPaliDuo, None, None]:
    device = get_torch_device("auto")
    logger.info(f"Device used: {device}")

    yield cast(
        ColPaliDuo,
        ColPaliDuo.from_pretrained(
            colpali_duo_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ),
    )
    tear_down_torch()


@pytest.fixture(scope="class")
def processor() -> Generator[ColPaliDuoProcessor, None, None]:
    yield cast(ColPaliDuoProcessor, ColPaliDuoProcessor.from_pretrained("google/paligemma-3b-mix-448"))


@pytest.fixture(scope="class")
def images() -> Generator[List[Image.Image], None, None]:
    yield [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (16, 16), color="black"),
    ]


@pytest.fixture(scope="class")
def queries() -> Generator[List[str], None, None]:
    yield [
        "Is attention all you need?",
        "What is the quantity of bananas farmed in Salvador?",
    ]


@pytest.fixture(scope="function")
def batch_queries(processor: ColPaliDuoProcessor, queries: List[str]) -> Generator[BatchFeature, None, None]:
    yield processor.process_queries(queries)


@pytest.fixture(scope="function")
def batch_images(processor: ColPaliDuoProcessor, images: List[Image.Image]) -> Generator[BatchFeature, None, None]:
    yield processor.process_images(images)


class TestLoadColPaliDuo:
    """
    Test the different ways to load ColPaliDuo.
    """

    @pytest.mark.slow
    def test_load_colpali_duo_from_config(self, colpali_duo_config: ColPaliDuoConfig):
        device = get_torch_device("auto")
        logger.info(f"Device used: {device}")

        model = ColPaliDuo(config=colpali_duo_config)

        assert isinstance(model, ColPaliDuo)
        assert model.single_vector_projector_dim == colpali_duo_config.single_vector_projector_dim
        assert model.multi_vector_pooler.pooling_strategy == colpali_duo_config.single_vector_pool_strategy
        assert model.multi_vector_projector_dim == colpali_duo_config.multi_vector_projector_dim

        tear_down_torch()

    @pytest.mark.slow
    def test_load_colpali_duo_from_pretrained(self, colpali_duo_from_pretrained: ColPaliDuo):
        assert isinstance(colpali_duo_from_pretrained, ColPaliDuo)


class TestColPaliDuoForwardSingleVector:
    """
    Test the forward pass of ColPaliDuo for single-vector embeddings.
    """

    @pytest.mark.slow
    def test_forward_images(
        self,
        test_load_colpali_duo_from_pretrained: ColPaliDuo,
        batch_images: BatchFeature,
    ):
        with torch.no_grad():
            outputs = test_load_colpali_duo_from_pretrained.forward_single_vector(**batch_images)

        assert isinstance(outputs.single_vec_emb, torch.Tensor)
        assert outputs.single_vec_emb.dim() == 2
        batch_size, emb_dim = outputs.single_vec_emb.shape
        assert batch_size == len(batch_images["input_ids"])
        assert emb_dim == test_load_colpali_duo_from_pretrained.single_vector_projector_dim

    @pytest.mark.slow
    def test_forward_queries(
        self,
        test_load_colpali_duo_from_pretrained: ColPaliDuo,
        batch_queries: BatchFeature,
    ):
        with torch.no_grad():
            outputs = test_load_colpali_duo_from_pretrained.forward_single_vector(**batch_queries)

        assert isinstance(outputs.single_vec_emb, torch.Tensor)
        assert outputs.single_vec_emb.dim() == 2
        batch_size, emb_dim = outputs.single_vec_emb.shape
        assert batch_size == len(batch_queries["input_ids"])
        assert emb_dim == test_load_colpali_duo_from_pretrained.single_vector_projector_dim


class TestColPaliDuoForwardMultiVector:
    """
    Test the forward pass of ColPaliDuo for multi-vector embeddings.
    """

    @pytest.mark.slow
    def test_forward_images(
        self,
        test_load_colpali_duo_from_pretrained: ColPaliDuo,
        batch_images: BatchFeature,
    ):
        with torch.no_grad():
            outputs = test_load_colpali_duo_from_pretrained.forward_multi_vector(**batch_images)

        assert isinstance(outputs.multi_vec_emb, torch.Tensor)
        assert outputs.multi_vec_emb.dim() == 3
        batch_size, n_visual_tokens, emb_dim = outputs.multi_vec_emb.shape
        assert batch_size == len(batch_images["input_ids"])
        assert emb_dim == test_load_colpali_duo_from_pretrained.multi_vector_projector_dim

    @pytest.mark.slow
    def test_forward_queries(
        self,
        test_load_colpali_duo_from_pretrained: ColPaliDuo,
        batch_queries: BatchFeature,
    ):
        with torch.no_grad():
            outputs = test_load_colpali_duo_from_pretrained.forward_multi_vector(**batch_queries)

        assert isinstance(outputs.multi_vec_emb, torch.Tensor)
        assert outputs.multi_vec_emb.dim() == 3
        batch_size, n_query_tokens, emb_dim = outputs.multi_vec_emb.shape
        assert batch_size == len(batch_queries["input_ids"])
        assert emb_dim == test_load_colpali_duo_from_pretrained.multi_vector_projector_dim


class TestColPaliDuoForwardAll:
    """
    Test the forward pass of ColPaliDuo (single-vector and multi-vector embeddings).
    """

    @pytest.mark.slow
    def test_forward_images(
        self,
        test_load_colpali_duo_from_pretrained: ColPaliDuo,
        batch_images: BatchFeature,
    ):
        with torch.no_grad():
            outputs = test_load_colpali_duo_from_pretrained.forward(**batch_images)

        assert isinstance(outputs, ColPaliDuoModelOutput)
        assert outputs.single_vec_emb is not None
        assert outputs.multi_vec_emb is not None
