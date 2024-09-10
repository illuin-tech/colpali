import pytest
import torch

from colpali_engine.evaluation.retrieval_scorer import RetrievalScorer


@pytest.fixture
def single_vector_scorer() -> RetrievalScorer:
    return RetrievalScorer(is_multi_vector=False, device="cpu")


@pytest.fixture
def multi_vector_scorer() -> RetrievalScorer:
    return RetrievalScorer(is_multi_vector=True, device="cpu")


def test_score_single_vector_embeddings(single_vector_scorer: RetrievalScorer):
    qs = [torch.randn(128) for _ in range(10)]
    ps = [torch.randn(128) for _ in range(10)]
    scores = single_vector_scorer.evaluate(qs, ps)
    assert scores.shape == (len(qs), len(ps))


def test_score_multi_vector_embeddings(multi_vector_scorer: RetrievalScorer):
    qs = [torch.randn(10, 128) for _ in range(10)]
    ps = [torch.randn(42, 128) for _ in range(10)]
    scores = multi_vector_scorer.evaluate(qs, ps)
    assert scores.shape == (len(qs), len(ps))
