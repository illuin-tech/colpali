import torch

from colpali_engine.interpretability.similarity_maps import normalize_similarity_map


def test_normalize_similarity_map_2d_ones():
    similarity_map = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    normalized_map = normalize_similarity_map(similarity_map)
    expected_map = torch.zeros_like(similarity_map)
    assert torch.allclose(normalized_map, expected_map, atol=1e-6)


def test_normalize_similarity_map_2d():
    similarity_map = torch.tensor(
        [
            [1.0, 1.0],
            [0.0, -1.0],
        ]
    )
    normalized_map = normalize_similarity_map(similarity_map)
    expected_map = torch.tensor(
        [
            [1.0, 1.0],
            [0.5, 0.0],
        ]
    )
    assert torch.allclose(normalized_map, expected_map, atol=1e-6)


def test_normalize_similarity_map_3d_ones():
    similarity_map = torch.tensor(
        [
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            [
                [2.0, 2.0],
                [2.0, 2.0],
            ],
        ]
    )
    normalized_map = normalize_similarity_map(similarity_map)
    expected_map = torch.zeros_like(similarity_map)
    assert torch.allclose(normalized_map, expected_map, atol=1e-6)
