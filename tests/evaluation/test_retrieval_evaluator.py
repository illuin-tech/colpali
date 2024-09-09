import pytest
import torch

from colpali_engine.evaluation.retrieval_evaluator import CustomRetrievalEvaluator


@pytest.fixture
def evaluator() -> CustomRetrievalEvaluator:
    return CustomRetrievalEvaluator(is_multi_vector=False, device="cpu")


@pytest.fixture
def multi_vector_evaluator() -> CustomRetrievalEvaluator:
    return CustomRetrievalEvaluator(is_multi_vector=True, device="cpu")


def test_evaluate_biencoder(evaluator: CustomRetrievalEvaluator):
    qs = [torch.randn(128) for _ in range(10)]
    ps = [torch.randn(128) for _ in range(10)]
    scores = evaluator.evaluate(qs, ps)
    assert scores.shape == (len(qs), len(ps))


def test_evaluate_colbert(multi_vector_evaluator: CustomRetrievalEvaluator):
    qs = [torch.randn(10, 128) for _ in range(10)]
    ps = [torch.randn(42, 128) for _ in range(10)]
    scores = multi_vector_evaluator.evaluate(qs, ps)
    assert scores.shape == (len(qs), len(ps))


@pytest.mark.skip("Not implemented")
def test_compute_metrics(evaluator: CustomRetrievalEvaluator):
    # TODO: Implement this test
    pass
