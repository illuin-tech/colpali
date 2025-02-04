from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
from scipy.cluster.hierarchy import fcluster, linkage


class BaseEmbeddingPooler(ABC):
    """
    Abstract class for pooling embeddings.
    """

    @abstractmethod
    def pool_embedding(
        self,
        embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor]]]:
        """
        Return the pooled embeddings and the mapping from cluster id to token indices.
        """
        pass


class HierarchicalEmbeddingPooler(BaseEmbeddingPooler):
    """
    Hierarchical pooling of embeddings based on the similarity between tokens.
    """

    def __init__(self, pool_factor: int):
        self.pool_factor = pool_factor

    def pool_embedding(
        self,
        embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor]]]:
        """
        Return the pooled embedding and the mapping from cluster id to token indices.

        This method doesn't support batched inputs because:
        - the sequence lengths can be different.
        - scipy doesn't support batched inputs.

        Args:
            embedding: A tensor of shape (token_length, embedding_dim).

        Returns:
            pooled_embeddings: A tensor of shape (num_clusters, embedding_dim).
            cluster_id_to_indices: A dictionary mapping cluster id to token indices.
        """
        list_pooled_embeddings: List[torch.Tensor] = []
        token_length = embedding.size(0)

        if token_length == 1:
            raise ValueError("The input tensor must have more than one token.")

        similarities = torch.mm(embedding, embedding.t())
        distances = 1 - similarities.to(torch.float32).cpu().numpy()

        Z = linkage(distances, metric="euclidean", method="ward")  # noqa: N806
        max_clusters = max(token_length // self.pool_factor, 1)
        cluster_labels = fcluster(Z, t=max_clusters, criterion="maxclust")

        cluster_id_to_indices: Dict[int, Tuple[torch.Tensor]] = {}

        with torch.no_grad():
            for cluster_id in range(1, max_clusters + 1):
                cluster_indices = torch.where(
                    torch.tensor(cluster_labels == cluster_id)
                )  # 1-tuple with tensor of shape (num_tokens,)
                cluster_id_to_indices[cluster_id] = cluster_indices

                if cluster_indices[0].numel() > 0:
                    pooled_embedding = embedding[cluster_indices].mean(dim=0)  # (embedding_dim,)
                    pooled_embedding = torch.nn.functional.normalize(pooled_embedding, p=2, dim=-1)  # (embedding_dim,)
                    list_pooled_embeddings.append(pooled_embedding)

            pooled_embeddings = torch.stack(list_pooled_embeddings, dim=0)  # (num_clusters, embedding_dim)

        return pooled_embeddings, cluster_id_to_indices
