import torch

from colpali_engine.utils.torch_utils import unbind_padded_multivector_embeddings


def test_unbind_padded_multivector_embedding_left():
    # Inputs:
    # Sequence 1: 4 tokens with 2 left-padding rows.
    seq1 = torch.tensor(
        [
            [0.0, 0.0],  # padding
            [0.0, 0.0],  # padding
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    # Sequence 2: 4 tokens with 1 left-padding row.
    seq2 = torch.tensor(
        [
            [0.0, 0.0],  # padding
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ],
        dtype=torch.float32,
    )
    padded_tensor = torch.stack([seq1, seq2], dim=0)  # shape: (2, 4, 2)

    # Call the unbind function.
    unbound = unbind_padded_multivector_embeddings(padded_tensor, padding_value=0.0, padding_side="left")

    # Expected outputs:
    expected_seq1 = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    expected_seq2 = torch.tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ],
        dtype=torch.float32,
    )

    assert len(unbound) == 2
    assert torch.allclose(unbound[0], expected_seq1)
    assert torch.allclose(unbound[1], expected_seq2)


def test_unbind_padded_multivector_embedding_right():
    # Inputs:
    # Sequence 1: valid tokens then padding.
    seq1 = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [0.0, 0.0],  # padding
        ],
        dtype=torch.float32,
    )
    # Sequence 2: valid tokens then padding.
    seq2 = torch.tensor(
        [
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [0.0, 0.0],  # padding
        ],
        dtype=torch.float32,
    )
    padded_tensor = torch.stack([seq1, seq2], dim=0)  # shape: (2, 4, 2)

    # Call the unbind function.
    unbound = unbind_padded_multivector_embeddings(padded_tensor, padding_value=0.0, padding_side="right")

    # Expected outputs:
    expected_seq1 = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    expected_seq2 = torch.tensor(
        [
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ],
        dtype=torch.float32,
    )

    assert len(unbound) == 2
    assert torch.allclose(unbound[0], expected_seq1)
    assert torch.allclose(unbound[1], expected_seq2)
