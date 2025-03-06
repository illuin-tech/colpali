import torch

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

EMBEDDING_DIM = 32


def test_score_single_vector_embeddings():
    qs = [torch.randn(EMBEDDING_DIM) for _ in range(4)]
    ps = [torch.randn(EMBEDDING_DIM) for _ in range(8)]
    scores = BaseVisualRetrieverProcessor.score_single_vector(qs, ps)
    assert scores.shape == (len(qs), len(ps))


def test_score_multi_vector_embeddings():
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
    scores_from_list_input = BaseVisualRetrieverProcessor.score_multi_vector(qs, ps)
    assert scores_from_list_input.shape == (len(qs), len(ps))

    # Score from tensor input
    qs_padded = torch.nn.utils.rnn.pad_sequence(qs, batch_first=True)
    ps_padded = torch.nn.utils.rnn.pad_sequence(ps, batch_first=True)
    scores_from_tensor = BaseVisualRetrieverProcessor.score_multi_vector(qs_padded, ps_padded)
    assert scores_from_tensor.shape == (len(qs), len(ps))

    assert torch.allclose(scores_from_list_input, scores_from_tensor), "Scores from list and tensor inputs should match"
