from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor  # New import for parallelism
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
    - pooled_embedding: A tensor of shape (num_clusters, embedding_dim).
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
        embeddings: Union[List[torch.Tensor], torch.Tensor],
        return_dict: bool = False,
    ) -> List[Union[torch.Tensor, TokenPoolingOutput]]:
        """
        Return the pooled embeddings.

        Args:
            embeddings: A list of 2D tensors (token_length, embedding_dim) where each tensor can have its own
                        token_length, or a 3D tensor of shape (batch_size, token_length, embedding_dim) without
                        padding.
            return_dict: Whether or not to return a `TokenPoolingOutput` object (with the cluster id to token indices
                         mapping) instead of just the pooled embeddings.

        Returns:
            A list of pooled embeddings or `TokenPoolingOutput` objects.
        """
        if isinstance(embeddings, list) and not embeddings:
            return []
        elif (isinstance(embeddings, list) and embeddings[0].dim() == 2) or (
            isinstance(embeddings, torch.Tensor) and embeddings.dim() == 3
        ):
            with ThreadPoolExecutor() as executor:
                # NOTE: We opted for a thread-based pool because most of the heavy lifting is done in C-level libraries
                # (NumPy, Torch, and SciPy) which usually release the GIL.
                results = list(executor.map(self._pool_single_embedding, embeddings, [return_dict] * len(embeddings)))
            return results
        else:
            raise ValueError("The input tensor must be a list of 2D tensors or a 3D tensor.")

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
            A pooled embedding tensor or a `TokenPoolingOutput` object.
        """
        if embedding.dim() != 2:
            raise ValueError("The input tensor must be a 2D tensor.")

        token_length = embedding.size(0)
        if token_length == 1:
            raise ValueError("The input tensor must have more than one token.")

        if self.pool_factor == 1:
            if not return_dict:
                return embedding
            return TokenPoolingOutput(
                pooled_embedding=embedding,
                cluster_id_to_indices={0: (torch.arange(token_length),)},
            )

        # Move the embedding to CPU for better multi-threading performance
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

        if not return_dict:
            return pooled_embeddings.to(device)

        return TokenPoolingOutput(
            pooled_embedding=pooled_embeddings.to(device),
            cluster_id_to_indices=cluster_id_to_indices,
        )
