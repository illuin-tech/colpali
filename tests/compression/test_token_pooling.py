import pytest
import torch

from colpali_engine.compression.token_pooling import HierarchicalTokenPooler, LambdaTokenPooler, TokenPoolingOutput


@pytest.fixture
def sample_embedding() -> torch.Tensor:
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
    )  # (6, 3)


class TestHierarchicalTokenPooler:
    def selftest_hierarchical_embedding_pooler_output_shape(self, sample_embedding: torch.Tensor):
        pooler = HierarchicalTokenPooler()
        outputs = pooler.pool_embeddings(
            [sample_embedding],
            pool_factor=2,
            return_dict=True,
        )

        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert isinstance(outputs[0], TokenPoolingOutput)
        assert outputs[0].pooled_embeddings.shape[1] == sample_embedding.shape[1]
        assert outputs[0].pooled_embeddings.shape[0] <= len(outputs[0].cluster_id_to_indices)

    def test_hierarchical_embedding_pooler_without_dict_output(self, sample_embedding: torch.Tensor):
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

    def test_hierarchical_embedding_pooler_with_dict_outputs(self, sample_embedding: torch.Tensor):
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

    def test_hierarchical_embedding_pooler_with_pool_factor_1(self, sample_embedding: torch.Tensor):
        pooler = HierarchicalTokenPooler()
        outputs = pooler.pool_embeddings(
            [sample_embedding],
            pool_factor=1,
            return_dict=True,
        )

        assert outputs.pooled_embeddings[0].shape == sample_embedding.shape
        assert torch.allclose(outputs.pooled_embeddings[0], sample_embedding)

    def test_hierarchical_embedding_pooler_with_different_pool_factors(self, sample_embedding: torch.Tensor):
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

    def test_hierarchical_embedding_pooler_should_raise_error_with_single_token(self):
        single_token_embeddings = torch.rand(1, 768)
        pooler = HierarchicalTokenPooler()

        with pytest.raises(ValueError):
            pooler.pool_embeddings(
                single_token_embeddings,
                pool_factor=2,
                return_dict=True,
            )

    def test_hierarchical_embedding_pooler_batched_input(self):
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

    def test_hierarchical_embedding_pooler_list_input(self):
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

    def test_pool_embeddings_padded_vs_list(self):
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


class TestLambdaTokenPooler:
    def test_lambda_token_pooler_initialization(self):
        def dummy_pool_func(embedding):
            return embedding, {0: (torch.arange(embedding.size(0)),)}

        pooler = LambdaTokenPooler(pool_func=dummy_pool_func)
        assert pooler.pool_func == dummy_pool_func

    def test_lambda_token_pooler_with_identity_function(self, sample_embedding: torch.Tensor):
        # Define a function that doesn't change the embeddings (identity function)
        def identity_pool_func(embedding):
            return embedding, {0: (torch.arange(embedding.size(0)),)}

        pooler = LambdaTokenPooler(pool_func=identity_pool_func)
        outputs = pooler.pool_embeddings([sample_embedding], return_dict=True)

        assert isinstance(outputs, TokenPoolingOutput)
        assert torch.allclose(outputs.pooled_embeddings[0], sample_embedding)
        assert len(outputs.cluster_id_to_indices[0]) == 1
        assert torch.allclose(outputs.cluster_id_to_indices[0][0][0], torch.arange(sample_embedding.size(0)))

    def test_lambda_token_pooler_with_mean_pooling(self, sample_embedding: torch.Tensor):
        # Define a function that pools all tokens into a single vector using mean
        def mean_pool_func(embedding):
            pooled_embedding = torch.mean(embedding, dim=0, keepdim=True)
            # Normalize the pooled embedding
            pooled_embedding = torch.nn.functional.normalize(pooled_embedding, p=2, dim=-1)
            return pooled_embedding, {0: (torch.arange(embedding.size(0)),)}

        pooler = LambdaTokenPooler(pool_func=mean_pool_func)
        outputs = pooler.pool_embeddings([sample_embedding], return_dict=True)

        assert isinstance(outputs, TokenPoolingOutput)
        assert outputs.pooled_embeddings[0].shape[0] == 1  # Single pooled vector
        assert outputs.pooled_embeddings[0].shape[1] == sample_embedding.shape[1]  # Same embedding dimension
        assert len(outputs.cluster_id_to_indices[0]) == 1
        assert torch.allclose(outputs.cluster_id_to_indices[0][0][0], torch.arange(sample_embedding.size(0)))

    def test_lambda_token_pooler_with_custom_pooling(self, sample_embedding: torch.Tensor):
        # Define a pooling function that groups tokens by similarity
        # This is a simple example that just groups by position (even/odd)
        def custom_pool_func(embedding):
            token_length = embedding.size(0)
            embedding_dim = embedding.size(1)

            # Group tokens into even and odd indices
            even_indices = torch.arange(0, token_length, 2)
            odd_indices = torch.arange(1, token_length, 2)

            # Create pooled embeddings
            pooled_embeddings = torch.zeros((2, embedding_dim), dtype=embedding.dtype, device=embedding.device)

            # Pool even indices
            if len(even_indices) > 0:
                pooled_embeddings[0] = torch.mean(embedding[even_indices], dim=0)
                pooled_embeddings[0] = torch.nn.functional.normalize(pooled_embeddings[0], p=2, dim=-1)

            # Pool odd indices
            if len(odd_indices) > 0:
                pooled_embeddings[1] = torch.mean(embedding[odd_indices], dim=0)
                pooled_embeddings[1] = torch.nn.functional.normalize(pooled_embeddings[1], p=2, dim=-1)

            # Return the mapping
            return pooled_embeddings, {
                0: (even_indices,),
                1: (odd_indices,),
            }

        pooler = LambdaTokenPooler(pool_func=custom_pool_func)
        outputs = pooler.pool_embeddings([sample_embedding], return_dict=True)

        assert isinstance(outputs, TokenPoolingOutput)
        assert outputs.pooled_embeddings[0].shape[0] == 2  # Two pooled vectors
        assert outputs.pooled_embeddings[0].shape[1] == sample_embedding.shape[1]  # Same embedding dimension
        assert len(outputs.cluster_id_to_indices[0]) == 2
        assert torch.allclose(outputs.cluster_id_to_indices[0][0][0], torch.arange(0, sample_embedding.size(0), 2))
        assert torch.allclose(outputs.cluster_id_to_indices[0][1][0], torch.arange(1, sample_embedding.size(0), 2))

    def test_lambda_token_pooler_with_batched_input(self):
        # Define a simple pooling function that reduces sequence length by half
        def halve_pool_func(embedding):
            token_length = embedding.size(0)
            half_length = token_length // 2 + (token_length % 2)
            pooled_embeddings = torch.zeros(
                (half_length, embedding.size(1)), dtype=embedding.dtype, device=embedding.device
            )

            cluster_id_to_indices = {}
            for i in range(half_length):
                start_idx = i * 2
                end_idx = min(start_idx + 2, token_length)
                cluster_indices = torch.arange(start_idx, end_idx)

                # Average the embeddings in the cluster
                pooled_embeddings[i] = embedding[cluster_indices].mean(dim=0)
                pooled_embeddings[i] = torch.nn.functional.normalize(pooled_embeddings[i], p=2, dim=-1)

                # Store mapping from cluster ID to token indices
                cluster_id_to_indices[i] = (cluster_indices,)

            return pooled_embeddings, cluster_id_to_indices

        # Create batch embeddings
        batch_embeddings = torch.rand(3, 10, 768)

        pooler = LambdaTokenPooler(pool_func=halve_pool_func)
        outputs = pooler.pool_embeddings(batch_embeddings, return_dict=True)

        assert isinstance(outputs, TokenPoolingOutput)
        # Expected output shape is (3, 5, 768) since we're halving the sequence length
        assert outputs.pooled_embeddings.shape == (3, 5, 768)

        # Check each example in the batch
        for i in range(3):
            # Verify cluster_id_to_indices structure
            assert len(outputs.cluster_id_to_indices[i]) == 5
            for cluster_id in range(5):
                if cluster_id < 4:
                    # Full clusters should have 2 elements
                    assert len(outputs.cluster_id_to_indices[i][cluster_id][0]) == 2
                else:
                    # Last cluster may have 1 or 2 elements
                    assert len(outputs.cluster_id_to_indices[i][cluster_id][0]) <= 2

    def test_lambda_token_pooler_with_list_input(self):
        # Define a simple pooling function that reduces sequence length by half
        def halve_pool_func(embedding):
            token_length = embedding.size(0)
            half_length = token_length // 2 + (token_length % 2)
            pooled_embeddings = torch.zeros(
                (half_length, embedding.size(1)), dtype=embedding.dtype, device=embedding.device
            )

            cluster_id_to_indices = {}
            for i in range(half_length):
                start_idx = i * 2
                end_idx = min(start_idx + 2, token_length)
                cluster_indices = torch.arange(start_idx, end_idx)

                # Average the embeddings in the cluster
                pooled_embeddings[i] = embedding[cluster_indices].mean(dim=0)
                pooled_embeddings[i] = torch.nn.functional.normalize(pooled_embeddings[i], p=2, dim=-1)

                # Store mapping from cluster ID to token indices
                cluster_id_to_indices[i] = (cluster_indices,)

            return pooled_embeddings, cluster_id_to_indices

        # Create list of embeddings with different lengths
        list_embeddings = [
            torch.rand(10, 768),
            torch.rand(15, 768),
        ]

        pooler = LambdaTokenPooler(pool_func=halve_pool_func)
        outputs = pooler.pool_embeddings(list_embeddings, return_dict=True)

        assert isinstance(outputs, TokenPoolingOutput)
        assert len(outputs.pooled_embeddings) == len(list_embeddings)

        # First embedding should be pooled to half size
        assert outputs.pooled_embeddings[0].shape[0] == 5
        assert outputs.pooled_embeddings[0].shape[1] == 768

        # Second embedding should be pooled to half size (rounded up)
        assert outputs.pooled_embeddings[1].shape[0] == 8
        assert outputs.pooled_embeddings[1].shape[1] == 768
