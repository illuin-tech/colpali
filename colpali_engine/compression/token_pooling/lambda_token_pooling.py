from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import torch

from colpali_engine.compression.token_pooling.base_token_pooling import BaseTokenPooler


class LambdaTokenPooler(BaseTokenPooler):
    """
    Token pooler that applies a user-defined pooling function to multi-vector embeddings.

    This pooler allows users to define custom pooling methods rather than relying on pre-defined pooling strategies.

    Example:

    ```python
    # Define a custom pooling function that reduces sequence length by half
    def custom_pooling(embedding: torch.Tensor) -> torch.Tensor:
        token_length = embedding.size(0)
        # Resize to half the original length by averaging pairs of tokens
        half_length = token_length // 2 + (token_length % 2)
        pooled_embeddings = torch.zeros(
            (half_length, embedding.size(1)),
            dtype=embedding.dtype,
            device=embedding.device,
        )

        for i in range(half_length):
            start_idx = i * 2
            end_idx = min(start_idx + 2, token_length)
            cluster_indices = torch.arange(start_idx, end_idx)
            pooled_embeddings[i] = embedding[cluster_indices].mean(dim=0)
            pooled_embeddings[i] = torch.nn.functional.normalize(pooled_embeddings[i], p=2, dim=-1)

        return pooled_embeddings


    # Create a LambdaTokenPooler with the custom function
    pooler = LambdaTokenPooler(pool_func=custom_pooling)
    outputs = pooler.pool_embeddings(embeddings)
    ```
    """

    def __init__(
        self,
        pool_func: Callable[[torch.Tensor], torch.Tensor],
    ):
        """
        Initialize the LambdaTokenPooler with a custom pooling function.

        Args:
            pool_func: A function that takes a 2D tensor (token_length, embedding_dim) and returns pooled embeddings,
                       i.e. a tensor of shape (num_clusters, embedding_dim)).
        """
        self.pool_func = pool_func

    def _pool_embeddings_impl(
        self,
        embeddings: List[torch.Tensor],
        num_workers: Optional[int] = None,
    ) -> Tuple[
        List[torch.Tensor],
        Optional[List[Dict[int, Tuple[torch.Tensor]]]],
    ]:
        """
        Apply the custom pooling function to each embedding in the list.

        Args:
            embeddings: List of 2D tensors to pool
            num_workers: Number of workers for parallel processing

        Returns:
            Tuple containing:
            - List of pooled embeddings
            - None (no cluster ID mapping in this implementation)
        """
        if num_workers and num_workers > 1:
            with ThreadPoolExecutor(num_workers) as executor:
                # NOTE: We opted for a thread-based pool because most of the heavy lifting is done in C-level libraries
                # (NumPy, Torch, and SciPy) which usually release the GIL.
                pooled_embeddings = list(executor.map(self.pool_func, embeddings))
        elif num_workers is None or num_workers == 1:
            # Process embeddings sequentially
            pooled_embeddings = [self.pool_func(emb) for emb in embeddings]
        else:
            raise ValueError(f"Invalid number of workers: {num_workers}")

        return pooled_embeddings, None
