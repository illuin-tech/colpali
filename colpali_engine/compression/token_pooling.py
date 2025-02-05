from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage


@dataclass
class TokenPoolingOutput:
    """
    Token pooling outputs:
    - pooled_embeddings: A tensor of shape (num_clusters, embedding_dim).
    - cluster_id_to_indices: A dictionary mapping cluster id to token indices.
    """

    pooled_embedding: torch.Tensor
    cluster_id_to_indices: Dict[int, Tuple[torch.Tensor]]


class BaseTokenPooler(ABC):
    """
    Abstract class for token pooling multi-vector embeddings.
    """

    @abstractmethod
    def pool_embeddings(
        self,
        embeddings: Union[torch.Tensor, List[torch.Tensor]],
    ) -> List[Union[torch.Tensor, TokenPoolingOutput]]:
        """
        Return the pooled multi-vector embeddings and the mapping from cluster id to token indices.
        """
        pass


class HierarchicalTokenPooler(BaseTokenPooler):
    """
    Hierarchical token pooling of multi-vector embeddings based on the similarity between tokens.
    """

    def __init__(self, pool_factor: int):
        """
        Args:
            pool_factor: An integer factor that determines the maximum number of clusters as
                         max_clusters = max(token_length // pool_factor, 1).
        """
        self.pool_factor = pool_factor

    def pool_embeddings(
        self,
        embeddings: Union[torch.Tensor, List[torch.Tensor]],
        return_dict: bool = False,
    ) -> List[Union[torch.Tensor, TokenPoolingOutput]]:
        """
        Return the pooled embeddings.

        The `embeddings` input can be one of the following:
        - A list of 2D tensors if the embeddings' sequence lengths are different.
        - A 3D tensor if the embeddings have the same length (no padding).

        Args:
            embeddings: A tensor of shape (token_length, embedding_dim) or (batch_size, token_length, embedding_dim).
            return_dict: Whether or not to return a `TokenPoolingOutput` object (with the cluster id to token indices
                         mapping) instead of just the pooled embeddings.

        Returns:
            A list of pooled embeddings or `PooledOutput` objects.
        """
        if isinstance(embeddings, torch.Tensor) and embeddings.dim() == 2:
            return [self._pool_single_embedding(embeddings, return_dict=return_dict)]
        elif isinstance(embeddings, list) or embeddings.dim() == 3:
            return [self._pool_single_embedding(batch_emb, return_dict) for batch_emb in embeddings]
        else:
            raise ValueError("The input tensor must be a list of 2D tensors or a 2D/3D tensor.")

    def _pool_single_embedding(
        self,
        embedding: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, TokenPoolingOutput]:
        """
        Return the pooled embedding and the mapping from cluster id to token indices.

        Args:
            embedding: A tensor of shape (token_length, embedding_dim).

        Returns:
            A list of pooled embeddings or `PooledOutput` objects.
        """
        if embedding.dim() != 2:
            raise ValueError("The input tensor must be a 2D tensor.")

        token_length = embedding.size(0)
        if token_length == 1:
            raise ValueError("The input tensor must have more than one token.")

        list_pooled_embeddings: List[torch.Tensor] = []

        similarities = torch.mm(embedding, embedding.t())
        distances = 1 - similarities.to(torch.float32).cpu().numpy()

        Z = linkage(distances, metric="euclidean", method="ward")  # noqa: N806
        max_clusters = max(token_length // self.pool_factor, 1)
        cluster_labels: NDArray[np.int32] = fcluster(Z, t=max_clusters, criterion="maxclust") - 1
        # NOTE: The scipy cluster labels start from 1, so we subtract 1 to start from 0.

        cluster_id_to_indices: Dict[int, Tuple[torch.Tensor]] = {}

        with torch.no_grad():
            for cluster_id in range(max_clusters):
                cluster_indices = cast(
                    Tuple[torch.Tensor],
                    torch.where(torch.tensor(cluster_labels == cluster_id)),
                )  # we know it is a 1-tuple
                cluster_id_to_indices[cluster_id] = cluster_indices

                if cluster_indices[0].numel() > 0:
                    pooled_embedding = embedding[cluster_indices].mean(dim=0)  # (embedding_dim,)
                    pooled_embedding = torch.nn.functional.normalize(pooled_embedding, p=2, dim=-1)  # (embedding_dim,)
                    list_pooled_embeddings.append(pooled_embedding)

            pooled_embeddings = torch.stack(list_pooled_embeddings, dim=0)  # (num_clusters, embedding_dim)

        if not return_dict:
            return pooled_embeddings

        return TokenPoolingOutput(
            pooled_embedding=pooled_embeddings,
            cluster_id_to_indices=cluster_id_to_indices,
        )
