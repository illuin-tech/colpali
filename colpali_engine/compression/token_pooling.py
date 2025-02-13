from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage

from colpali_engine.utils.torch_utils import unbind_padded_multivector_embeddings


@dataclass
class TokenPoolingOutput:
    """
    Token pooling outputs:
    - pooled_embeddings: A list of 2D tensors (token_length, embedding_dim) where each tensor can have its own
                        token_length, or a 3D tensor of shape (batch_size, token_length, embedding_dim) with
                        optional padding.
    - cluster_id_to_indices: A list of dictionaries. The i-th dictionary maps the cluster id to token indices for
                             the i-th embedding in `pooled_embeddings`.
    """

    pooled_embeddings: Union[List[torch.Tensor], torch.Tensor]
    cluster_id_to_indices: List[Dict[int, Tuple[torch.Tensor]]]


class BaseTokenPooler(ABC):
    """
    Abstract class for token pooling multi-vector embeddings.
    """

    @abstractmethod
    def pool_embeddings(
        self,
        embeddings: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Union[Union[torch.Tensor, List[torch.Tensor]], TokenPoolingOutput]:
        """
        Return the pooled multi-vector embeddings and the mapping from cluster id to token indices.
        """
        pass


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

    def pool_embeddings(
        self,
        embeddings: Union[List[torch.Tensor], torch.Tensor],
        return_dict: bool = False,
        padding: bool = False,
        padding_side: str = "left",
        num_workers: Optional[int] = None,
    ) -> Union[Union[torch.Tensor, List[torch.Tensor]], TokenPoolingOutput]:
        """
        Return the pooled embeddings.

        Args:
            embeddings: A list of 2D tensors (token_length, embedding_dim) where each tensor can have its own token
                        length, or a 3D tensor of shape (batch_size, token_length, embedding_dim) with 0-padding.
            return_dict: Whether or not to return a `TokenPoolingOutput` object (with the cluster id to token indices
                         mapping) instead of just the pooled embeddings.
            padding: Whether or not to unbind the padded 3D tensor into a list of 2D tensors. Does nothing if the input
                     is a list of 2D tensors.
            padding_side: The side where the padding was applied in the 3D tensor.

        Returns:
            If the `embeddings` input is:
            - A list of 2D tensors: Returns a list of 2D tensors (token_length, embedding_dim) where each tensor can
                                    have its own token_length.
            - A 3D tensor: A 3D tensor of shape (batch_size, token_length, embedding_dim) with 0-padding.

            If `return_dict` is True, the pooled embeddings are returned within a `TokenPoolingOutput` object, along
            with the cluster id to token indices mapping.
        """
        if isinstance(embeddings, list) and not embeddings:
            return TokenPoolingOutput(pooled_embeddings=[], cluster_id_to_indices=[])

        is_list_of_2d_tensors = isinstance(embeddings, list) and embeddings[0].dim() == 2
        is_3d_tensor = isinstance(embeddings, torch.Tensor) and embeddings.dim() == 3

        if not is_list_of_2d_tensors and not is_3d_tensor:
            raise ValueError("The input tensor must be a list of 2D tensors or a 3D tensor.")

        if is_3d_tensor:
            if padding:
                embeddings = unbind_padded_multivector_embeddings(
                    embeddings,
                    padding_value=0.0,
                    padding_side=padding_side,
                )
            else:
                embeddings = list(embeddings.unbind(dim=0))

        with ThreadPoolExecutor(num_workers) as executor:
            # NOTE: We opted for a thread-based pool because most of the heavy lifting is done in C-level libraries
            # (NumPy, Torch, and SciPy) which usually release the GIL.
            results = list(executor.map(self._pool_single_embedding, embeddings))

        # Unpack the results
        pooled_embeddings = [result[0] for result in results]
        cluster_id_to_indices = [result[1] for result in results]

        if is_3d_tensor:
            # Repad the pooled embeddings
            pooled_embeddings = torch.nn.utils.rnn.pad_sequence(
                pooled_embeddings,
                batch_first=True,
                padding_value=0.0,
                padding_side=padding_side,
            )

        if not return_dict:
            return pooled_embeddings

        return TokenPoolingOutput(
            pooled_embeddings=pooled_embeddings,
            cluster_id_to_indices=cluster_id_to_indices,
        )

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
