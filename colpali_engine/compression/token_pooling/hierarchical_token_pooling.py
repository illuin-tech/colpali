from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage

from colpali_engine.compression.token_pooling.base_token_pooling import BaseTokenPooler


class HierarchicalTokenPooler(BaseTokenPooler):
    """
    Hierarchical token pooling of multi-vector embeddings based on the similarity between token embeddings.

    Example with a list of 2D tensors:

    ```python
    list_embeddings = [torch.rand(10, 768), torch.rand(20, 768)]
    pooler = HierarchicalTokenPooler(pool_factor=2)
    outputs = pooler.pool_embeddings(list_embeddings)
    ```

    Example with a 0-padded 3D tensor:

    ```python
    list_embeddings = [torch.rand(10, 768), torch.rand(20, 768)]
    padded_embeddings = torch.nn.utils.rnn.pad_sequence(
            list_embeddings,
            batch_first=True,
            padding_value=0.0,
            padding_side="left",
        )
    pooler = HierarchicalTokenPooler(pool_factor=2)
    outputs = pooler.pool_embeddings(list_embeddings, padding=True, padding_side="left")
    ```
    """

    def __init__(self, pool_factor: int):
        """
        Args:
            pool_factor: An integer factor that determines the maximum number of clusters as
                         `max_clusters = max(token_length // pool_factor, 1)`.
        """
        self.pool_factor = pool_factor

    def _pool_embeddings_impl(
        self,
        embeddings: List[torch.Tensor],
        num_workers: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[Dict[int, Tuple[torch.Tensor]]]]:
        """
        Apply hierarchical pooling to each embedding in the list.

        Args:
            embeddings: List of 2D tensors to pool
            num_workers: Number of workers for parallel processing

        Returns:
            Tuple containing:
            - List of pooled embeddings
            - List of dictionaries mapping cluster IDs to token indices
        """
        with ThreadPoolExecutor(num_workers) as executor:
            # NOTE: We opted for a thread-based pool because most of the heavy lifting is done in C-level libraries
            # (NumPy, Torch, and SciPy) which usually release the GIL.
            results = list(executor.map(self._pool_single_embedding, embeddings))

        # Unpack the results
        pooled_embeddings = [result[0] for result in results]
        cluster_id_to_indices = [result[1] for result in results]

        return pooled_embeddings, cluster_id_to_indices

    def _pool_single_embedding(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor]]]:
        """
        Return the pooled embedding and the mapping from cluster id to token indices.

        Args:
            embedding: A tensor of shape (token_length, embedding_dim).

        Returns:
            pooled_embedding: A tensor of shape (num_clusters, embedding_dim).
            cluster_id_to_indices: A dictionary mapping the cluster id to token indices.
        """
        if embedding.dim() != 2:
            raise ValueError("The input tensor must be a 2D tensor.")

        token_length = embedding.size(0)
        if token_length == 1:
            raise ValueError("The input tensor must have more than one token.")

        if self.pool_factor == 1:
            cluster_id_to_indices = {0: (torch.arange(token_length),)}
            return embedding, cluster_id_to_indices

        # Move the embedding to CPU for better multi-threading performance
        dtype = embedding.dtype
        device = embedding.device
        embedding = embedding.to(torch.float32).cpu()

        list_pooled_embeddings: List[torch.Tensor] = []

        similarities = torch.mm(embedding, embedding.t())
        distances = 1 - similarities.numpy()

        Z = linkage(distances, metric="euclidean", method="ward")  # noqa: N806
        max_clusters = max(token_length // self.pool_factor, 1)
        cluster_labels: NDArray[np.int32] = fcluster(Z, t=max_clusters, criterion="maxclust") - 1
        # NOTE: The scipy cluster labels start from 1, so we subtract 1 to start from 0.

        cluster_id_to_indices: Dict[int, Tuple[torch.Tensor]] = {}

        with torch.no_grad():
            for cluster_id in range(max_clusters):
                cluster_indices = cast(
                    Tuple[torch.Tensor],  # we know it is a 1-tuple
                    torch.where(torch.tensor(cluster_labels == cluster_id)),
                )
                cluster_id_to_indices[cluster_id] = cluster_indices

                if cluster_indices[0].numel() > 0:
                    pooled_embedding = embedding[cluster_indices].mean(dim=0)  # (embedding_dim,)
                    pooled_embedding = torch.nn.functional.normalize(pooled_embedding, p=2, dim=-1)
                    list_pooled_embeddings.append(pooled_embedding)

            pooled_embeddings = torch.stack(list_pooled_embeddings, dim=0)  # (num_clusters, embedding_dim)

        # Restore the original device and dtype
        pooled_embeddings = pooled_embeddings.to(device).to(dtype)

        return pooled_embeddings, cluster_id_to_indices
