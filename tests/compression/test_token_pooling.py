import pytest
import torch

from colpali_engine.compression.token_pooling import HierarchicalTokenPooler, TokenPoolingOutput


@pytest.fixture
def sample_embeddings() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )


def test_hierarchical_embedding_pooler_initialization():
    pooler = HierarchicalTokenPooler(pool_factor=2)
    assert pooler.pool_factor == 2


def test_hierarchical_embedding_pooler_output_shape(sample_embeddings: torch.Tensor):
    pooler = HierarchicalTokenPooler(pool_factor=2)
    outputs = pooler.pool_embeddings(sample_embeddings, return_dict=True)

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], TokenPoolingOutput)
    assert outputs[0].pooled_embedding.shape[1] == sample_embeddings.shape[1]
    assert outputs[0].pooled_embedding.shape[0] <= len(outputs[0].cluster_id_to_indices)


def test_hierarchical_embedding_pooler_without_dict_output(sample_embeddings: torch.Tensor):
    pooler = HierarchicalTokenPooler(pool_factor=2)
    outputs = pooler.pool_embeddings(sample_embeddings, return_dict=False)

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], torch.Tensor)

    expected_pooled_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(outputs[0], expected_pooled_embeddings)


def test_hierarchical_embedding_pooler_output_values(sample_embeddings: torch.Tensor):
    pooler = HierarchicalTokenPooler(pool_factor=2)
    outputs = pooler.pool_embeddings(sample_embeddings, return_dict=True)

    expected_pooled_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    expected_cluster_id_to_indices = {
        0: (torch.tensor([0, 3, 4, 5]),),
        1: (torch.tensor([1]),),
        2: (torch.tensor([2]),),
    }

    assert torch.allclose(outputs[0].pooled_embedding, expected_pooled_embeddings)
    assert all(
        [
            torch.allclose(outputs[0].cluster_id_to_indices[cluster_id][0], expected_cluster_indices[0])
            for cluster_id, expected_cluster_indices in expected_cluster_id_to_indices.items()
        ]
    )


def test_hierarchical_embedding_pooler_with_different_pool_factors(sample_embeddings: torch.Tensor):
    for pool_factor in [1, 2, 3]:
        pooler = HierarchicalTokenPooler(pool_factor=pool_factor)
        outputs = pooler.pool_embeddings(sample_embeddings, return_dict=True)
        expected_num_clusters = (sample_embeddings.shape[0] + pool_factor - 1) // pool_factor
        assert outputs[0].pooled_embedding.shape[0] <= expected_num_clusters
        assert outputs[0].pooled_embedding.shape[0] <= len(outputs[0].cluster_id_to_indices)


def test_hierarchical_embedding_pooler_should_raise_error_with_single_token():
    single_token_embeddings = torch.rand(1, 768)
    pooler = HierarchicalTokenPooler(pool_factor=2)

    with pytest.raises(ValueError):
        pooler.pool_embeddings(single_token_embeddings, return_dict=True)


def test_hierarchical_embedding_pooler_batched_input():
    batch_embeddings = torch.rand(3, 10, 768)

    pooler = HierarchicalTokenPooler(pool_factor=2)
    outputs = pooler.pool_embeddings(batch_embeddings, return_dict=True)

    assert len(outputs) == 3
    for output in outputs:
        assert isinstance(output, TokenPoolingOutput)
        assert output.pooled_embedding.shape[1] == 768
        assert output.pooled_embedding.shape[0] <= 5


def test_hierarchical_embedding_pooler_list_input():
    list_embeddings = [
        torch.rand(10, 768),
        torch.rand(15, 768),
    ]

    pooler = HierarchicalTokenPooler(pool_factor=2)
    outputs = pooler.pool_embeddings(list_embeddings, return_dict=True)

    assert len(outputs) == len(list_embeddings)
    for input_embedding, output in zip(list_embeddings, outputs):
        assert isinstance(output, TokenPoolingOutput)
        assert output.pooled_embedding.shape[1] == 768
        assert output.pooled_embedding.shape[0] <= input_embedding.shape[0] // 2
