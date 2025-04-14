import torch


def identity_pool_func(embedding: torch.Tensor) -> torch.Tensor:
    """
    Returns the unchanged embeddings.

    Args:
        embedding (torch.Tensor): The embeddings to pool (token_length, embedding_dim).

    Returns:
        torch.Tensor: The unchanged embeddings (token_length, embedding_dim).
    """
    return embedding  # (token_length, embedding_dim)


def halve_pool_func(embedding: torch.Tensor) -> torch.Tensor:
    """
    Pools the embeddings by averaging all pairs of consecutive vectors.
    Resulting embedding dimension is half the original dimension.

    Args:
        embedding (torch.Tensor): The embeddings to pool (token_length, embedding_dim).

    Returns:
        torch.Tensor: The pooled embeddings (half_length, embedding_dim).
    """
    token_length = embedding.size(0)
    half_length = token_length // 2 + (token_length % 2)

    pooled_embeddings = torch.zeros(
        (half_length, embedding.size(1)),
        dtype=embedding.dtype,
        device=embedding.device,
    )

    for i in range(half_length):
        start_idx = i * 2
        end_idx = min(start_idx + 2, token_length)
        cluster_indices = torch.arange(start_idx, end_idx)  # (2)
        pooled_embeddings[i] = embedding[cluster_indices].mean(dim=0)  # (embedding_dim)

    return pooled_embeddings  # (half_length, embedding_dim)


def mean_pool_func(embedding: torch.Tensor) -> torch.Tensor:
    """
    Pools the embeddings by averaging all the embeddings in the cluster.

    Args:
        embedding (torch.Tensor): The embeddings to pool (token_length, embedding_dim).

    Returns:
        torch.Tensor: The pooled embeddings (1, embedding_dim).
    """
    pooled_embedding = torch.mean(embedding, dim=0, keepdim=True)  # (1, embedding_dim)
    pooled_embedding = torch.nn.functional.normalize(pooled_embedding, p=2, dim=-1)  # (1, embedding_dim)
    return pooled_embedding  # (1, embedding_dim)
