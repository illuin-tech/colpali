import pytest
import torch

from colpali_engine.compression.token_pooling import HierarchicalTokenPooler, TokenPoolingOutput


def test_hierarchical_embedding_pooler_output_shape(sample_embedding: torch.Tensor):
    pooler = HierarchicalTokenPooler()
    outputs = pooler.pool_embeddings(
        [sample_embedding],
        pool_factor=2,
        return_dict=True,
    )

    assert isinstance(outputs, TokenPoolingOutput)
    assert len(outputs.pooled_embeddings) == 1
    assert outputs.pooled_embeddings[0].shape[1] == sample_embedding.shape[1]

    assert outputs.cluster_id_to_indices is not None
    assert outputs.pooled_embeddings[0].shape[0] <= len(outputs.cluster_id_to_indices[0])


def test_hierarchical_embedding_pooler_without_dict_output(sample_embedding: torch.Tensor):
    pooler = HierarchicalTokenPooler()
    outputs = pooler.pool_embeddings(
        [sample_embedding],
        pool_factor=2,
        return_dict=False,
    )

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


def test_hierarchical_embedding_pooler_with_dict_outputs(sample_embedding: torch.Tensor):
    pooler = HierarchicalTokenPooler()
    outputs = pooler.pool_embeddings(
        [sample_embedding],
        pool_factor=2,
        return_dict=True,
    )

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

    assert torch.allclose(outputs.pooled_embeddings[0], expected_pooled_embeddings)
    assert all(
        [
            torch.allclose(outputs.cluster_id_to_indices[0][cluster_id][0], expected_cluster_indices[0])
            for cluster_id, expected_cluster_indices in expected_cluster_id_to_indices.items()
        ]
    )


def test_hierarchical_embedding_pooler_with_pool_factor_1(sample_embedding: torch.Tensor):
    pooler = HierarchicalTokenPooler()
    outputs = pooler.pool_embeddings(
        [sample_embedding],
        pool_factor=1,
        return_dict=True,
    )

    assert outputs.pooled_embeddings[0].shape == sample_embedding.shape
    assert torch.allclose(outputs.pooled_embeddings[0], sample_embedding)


def test_hierarchical_embedding_pooler_with_different_pool_factors(sample_embedding: torch.Tensor):
    pooler = HierarchicalTokenPooler()

    for pool_factor in range(1, 6):
        outputs = pooler.pool_embeddings(
            [sample_embedding],
            pool_factor=pool_factor,
            return_dict=True,
        )

        expected_num_clusters = max(sample_embedding.shape[0] // pool_factor, 1)

        assert outputs.pooled_embeddings[0].shape[0] == expected_num_clusters
        assert outputs.pooled_embeddings[0].shape[0] <= len(outputs.cluster_id_to_indices[0][0][0])


def test_hierarchical_embedding_pooler_should_raise_error_with_single_token():
    single_token_embeddings = torch.rand(1, 768)
    pooler = HierarchicalTokenPooler()

    with pytest.raises(ValueError):
        pooler.pool_embeddings(
            single_token_embeddings,
            pool_factor=2,
            return_dict=True,
        )


def test_hierarchical_embedding_pooler_batched_input():
    batch_embeddings = torch.rand(3, 10, 768)

    pooler = HierarchicalTokenPooler()
    outputs = pooler.pool_embeddings(
        batch_embeddings,
        pool_factor=2,
        return_dict=True,
    )

    assert isinstance(outputs, TokenPoolingOutput)
    for pooled_embedding in outputs.pooled_embeddings:
        assert pooled_embedding.shape[1] == 768
        assert pooled_embedding.shape[0] <= 5


def test_hierarchical_embedding_pooler_list_input():
    list_embeddings = [
        torch.rand(10, 768),
        torch.rand(15, 768),
    ]

    pooler = HierarchicalTokenPooler()
    outputs = pooler.pool_embeddings(
        list_embeddings,
        pool_factor=2,
        return_dict=True,
    )

    assert len(outputs.pooled_embeddings) == len(list_embeddings)
    for input_embedding, pooled_embedding in zip(list_embeddings, outputs.pooled_embeddings):
        assert pooled_embedding.shape[1] == 768
        assert pooled_embedding.shape[0] <= input_embedding.shape[0] // 2


def test_pool_embeddings_padded_vs_list():
    pooler = HierarchicalTokenPooler()

    # Reference.
    seq1 = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    seq2 = torch.tensor(
        [
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
        ],
        dtype=torch.float32,
    )
    list_embeddings = [seq1, seq2]

    list_pooled_embeddings = pooler.pool_embeddings(
        list_embeddings,
        pool_factor=1,
        return_dict=False,
    )
    expected_pooled_embeddings = torch.nn.utils.rnn.pad_sequence(
        list_pooled_embeddings,
        batch_first=True,
        padding_value=0.0,
        padding_side="left",
    )

    # Pad the reference embeddngs.
    padded_embeddings = torch.nn.utils.rnn.pad_sequence(
        list_embeddings,
        batch_first=True,
        padding_value=0.0,
        padding_side="left",
    )

    # Call pool_embeddings with the padded tensor. Note that we must pass padding=True.
    pooled_embeddings = pooler.pool_embeddings(
        padded_embeddings,
        pool_factor=1,
        return_dict=False,
        padding=True,
        padding_side="left",
    )

    assert torch.allclose(pooled_embeddings, expected_pooled_embeddings)
