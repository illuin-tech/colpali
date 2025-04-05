import torch

from colpali_engine.compression.token_pooling import LambdaTokenPooler, TokenPoolingOutput


def test_lambda_token_pooler_initialization():
    def dummy_pool_func(embedding):
        return embedding  # Remove the second return value

    pooler = LambdaTokenPooler(pool_func=dummy_pool_func)
    assert pooler.pool_func == dummy_pool_func


def test_lambda_token_pooler_with_identity_function(sample_embedding: torch.Tensor):
    # Define a function that doesn't change the embeddings (identity function)
    def identity_pool_func(embedding):
        return embedding  # Remove the second return value

    pooler = LambdaTokenPooler(pool_func=identity_pool_func)
    outputs = pooler.pool_embeddings([sample_embedding], return_dict=True)

    assert isinstance(outputs, TokenPoolingOutput)
    assert torch.allclose(outputs.pooled_embeddings[0], sample_embedding)
    # Cluster mapping is now None
    assert outputs.cluster_id_to_indices is None


def test_lambda_token_pooler_with_mean_pooling(sample_embedding: torch.Tensor):
    # Define a function that pools all tokens into a single vector using mean
    def mean_pool_func(embedding):
        pooled_embedding = torch.mean(embedding, dim=0, keepdim=True)
        # Normalize the pooled embedding
        pooled_embedding = torch.nn.functional.normalize(pooled_embedding, p=2, dim=-1)
        return pooled_embedding  # Remove the second return value

    pooler = LambdaTokenPooler(pool_func=mean_pool_func)
    outputs = pooler.pool_embeddings([sample_embedding], return_dict=True)

    assert isinstance(outputs, TokenPoolingOutput)
    assert outputs.pooled_embeddings[0].shape[0] == 1  # Single pooled vector
    assert outputs.pooled_embeddings[0].shape[1] == sample_embedding.shape[1]  # Same embedding dimension
    # Cluster mapping is now None
    assert outputs.cluster_id_to_indices is None


def test_lambda_token_pooler_with_custom_pooling(sample_embedding: torch.Tensor):
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

        # Return only the pooled embeddings
        return pooled_embeddings

    pooler = LambdaTokenPooler(pool_func=custom_pool_func)
    outputs = pooler.pool_embeddings([sample_embedding], return_dict=True)

    assert isinstance(outputs, TokenPoolingOutput)
    assert outputs.pooled_embeddings[0].shape[0] == 2  # Two pooled vectors
    assert outputs.pooled_embeddings[0].shape[1] == sample_embedding.shape[1]  # Same embedding dimension
    # Cluster mapping is now None
    assert outputs.cluster_id_to_indices is None


def test_lambda_token_pooler_with_batched_input():
    # Define a simple pooling function that reduces sequence length by half
    def halve_pool_func(embedding):
        token_length = embedding.size(0)
        half_length = token_length // 2 + (token_length % 2)
        pooled_embeddings = torch.zeros(
            (half_length, embedding.size(1)), dtype=embedding.dtype, device=embedding.device
        )

        for i in range(half_length):
            start_idx = i * 2
            end_idx = min(start_idx + 2, token_length)
            cluster_indices = torch.arange(start_idx, end_idx)

            # Average the embeddings in the cluster
            pooled_embeddings[i] = embedding[cluster_indices].mean(dim=0)
            pooled_embeddings[i] = torch.nn.functional.normalize(pooled_embeddings[i], p=2, dim=-1)

        return pooled_embeddings  # Remove the second return value

    # Create batch embeddings
    batch_embeddings = torch.rand(3, 10, 768)

    pooler = LambdaTokenPooler(pool_func=halve_pool_func)
    outputs = pooler.pool_embeddings(batch_embeddings, return_dict=True)

    assert isinstance(outputs, TokenPoolingOutput)
    # Expected output shape is (3, 5, 768) since we're halving the sequence length
    assert isinstance(outputs.pooled_embeddings, torch.Tensor)  # Should be a tensor for batched input
    assert outputs.pooled_embeddings.shape == (3, 5, 768)

    # Cluster mapping is now None
    assert outputs.cluster_id_to_indices is None


def test_lambda_token_pooler_with_list_input():
    # Define a simple pooling function that reduces sequence length by half
    def halve_pool_func(embedding):
        token_length = embedding.size(0)
        half_length = token_length // 2 + (token_length % 2)
        pooled_embeddings = torch.zeros(
            (half_length, embedding.size(1)), dtype=embedding.dtype, device=embedding.device
        )

        for i in range(half_length):
            start_idx = i * 2
            end_idx = min(start_idx + 2, token_length)
            cluster_indices = torch.arange(start_idx, end_idx)

            # Average the embeddings in the cluster
            pooled_embeddings[i] = embedding[cluster_indices].mean(dim=0)
            pooled_embeddings[i] = torch.nn.functional.normalize(pooled_embeddings[i], p=2, dim=-1)

        return pooled_embeddings  # Remove the second return value

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

    # Cluster mapping is now None
    assert outputs.cluster_id_to_indices is None
