import pytest
import torch

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


@pytest.fixture
def vector_scorer() -> BaseVisualRetrieverProcessor:
    return BaseVisualRetrieverProcessor()


def test_score_single_vector_embeddings(vector_scorer: BaseVisualRetrieverProcessor):
    qs = [torch.randn(128) for _ in range(10)]
    ps = [torch.randn(128) for _ in range(10)]
    scores = vector_scorer.score_single_vector(qs, ps)
    assert scores.shape == (len(qs), len(ps))


def test_score_multi_vector_embeddings(vector_scorer: BaseVisualRetrieverProcessor):
    qs = [torch.randn(10, 128) for _ in range(10)]
    ps = [torch.randn(42, 128) for _ in range(10)]
    scores = vector_scorer.score_multi_vector(qs, ps)
    assert scores.shape == (len(qs), len(ps))
