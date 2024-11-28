from unittest.mock import patch

import pytest
import torch

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

EMBEDDING_DIM = 32


@pytest.fixture
@patch.multiple(BaseVisualRetrieverProcessor, __abstractmethods__=set())
def processor() -> BaseVisualRetrieverProcessor:
    return BaseVisualRetrieverProcessor()  # type: ignore


def test_score_single_vector_embeddings(processor: BaseVisualRetrieverProcessor):
    qs = [torch.randn(EMBEDDING_DIM) for _ in range(4)]
    ps = [torch.randn(EMBEDDING_DIM) for _ in range(8)]
    scores = processor.score_single_vector(qs, ps)
    assert scores.shape == (len(qs), len(ps))


def test_score_multi_vector_embeddings(processor: BaseVisualRetrieverProcessor):
    # Score from list input
    qs = [
        torch.randn(2, EMBEDDING_DIM),
        torch.randn(4, EMBEDDING_DIM),
    ]
    ps = [
        torch.randn(8, EMBEDDING_DIM),
        torch.randn(4, EMBEDDING_DIM),
        torch.randn(16, EMBEDDING_DIM),
    ]
    scores_from_list_input = processor.score_multi_vector(qs, ps)
    assert scores_from_list_input.shape == (len(qs), len(ps))

    # Score from tensor input
    qs_padded = torch.nn.utils.rnn.pad_sequence(qs, batch_first=True)
    ps_padded = torch.nn.utils.rnn.pad_sequence(ps, batch_first=True)
    scores_from_tensor = processor.score_multi_vector(qs_padded, ps_padded)
    assert scores_from_tensor.shape == (len(qs), len(ps))

    assert torch.allclose(scores_from_list_input, scores_from_tensor), "Scores from list and tensor inputs should match"
