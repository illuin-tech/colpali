from unittest.mock import patch

import pytest
import torch

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


@pytest.fixture
@patch.multiple(BaseVisualRetrieverProcessor, __abstractmethods__=set())
def processor() -> BaseVisualRetrieverProcessor:
    return BaseVisualRetrieverProcessor()  # type: ignore


def test_score_single_vector_embeddings(processor: BaseVisualRetrieverProcessor):
    qs = [torch.randn(128) for _ in range(10)]
    ps = [torch.randn(128) for _ in range(10)]
    scores = processor.score_single_vector(qs, ps)
    assert scores.shape == (len(qs), len(ps))


def test_score_multi_vector_embeddings(processor: BaseVisualRetrieverProcessor):
    qs = [torch.randn(10, 128) for _ in range(10)]
    ps = [torch.randn(42, 128) for _ in range(10)]
    scores = processor.score_multi_vector(qs, ps)
    assert scores.shape == (len(qs), len(ps))
