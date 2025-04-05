from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

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
    cluster_id_to_indices: Optional[List[Dict[int, Tuple[torch.Tensor]]]] = None


class BaseTokenPooler(ABC):
    """
    Abstract class for token pooling multi-vector embeddings.
    """

    @abstractmethod
    def _pool_embeddings_impl(
        self,
        embeddings: List[torch.Tensor],
        num_workers: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], List[Dict[int, Tuple[torch.Tensor]]]]:
        """
        Implementation of pooling logic for a list of 2D embeddings.

        Args:
            embeddings: A list of 2D tensors (token_length, embedding_dim)
            num_workers: Number of workers for parallel processing

        Returns:
            Tuple containing:
            - List of pooled embeddings
            - List of dictionaries mapping cluster IDs to token indices
        """
        pass

    def _validate_embeddings(self, embeddings: Union[List[torch.Tensor], torch.Tensor]) -> None:
        """
        Validate input embeddings and determine their type.

        Args:
            embeddings: Input embeddings to validate

        Raises:
            ValueError: If embeddings are empty or have invalid dimensions
        """
        if isinstance(embeddings, list) and not embeddings:
            raise ValueError("Empty embeddings list provided")

        is_list_of_2d_tensors = isinstance(embeddings, list) and embeddings[0].dim() == 2
        is_3d_tensor = isinstance(embeddings, torch.Tensor) and embeddings.dim() == 3

        if not is_list_of_2d_tensors and not is_3d_tensor:
            raise ValueError("The input tensor must be a list of 2D tensors or a 3D tensor.")

    def _prepare_embeddings(
        self,
        embeddings: Union[List[torch.Tensor], torch.Tensor],
        padding: bool = False,
        padding_side: str = "left",
    ) -> List[torch.Tensor]:
        """
        Prepare embeddings for pooling by converting to a list of 2D tensors.

        Args:
            embeddings: Input embeddings
            padding: Whether to unbind padded 3D tensor
            padding_side: Side where padding was applied

        Returns:
            List of 2D tensors ready for pooling
        """
        is_3d_tensor = isinstance(embeddings, torch.Tensor) and embeddings.dim() == 3
        if is_3d_tensor:
            if padding:
                return unbind_padded_multivector_embeddings(
                    embeddings=cast(torch.Tensor, embeddings),
                    padding_value=0.0,
                    padding_side=padding_side,
                )
            else:
                return list(cast(torch.Tensor, embeddings).unbind(dim=0))

        return cast(List[torch.Tensor], embeddings)

    def pool_embeddings(
        self,
        embeddings: Union[torch.Tensor, List[torch.Tensor]],
        return_dict: bool = False,
        padding: bool = False,
        padding_side: str = "left",
        num_workers: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Union[Union[torch.Tensor, List[torch.Tensor]], TokenPoolingOutput]:
        """
        Return the pooled multi-vector embeddings and the mapping from cluster id to token indices.

        Args:
            embeddings: A list of 2D tensors (token_length, embedding_dim) where each tensor can have its own token
                        length, or a 3D tensor of shape (batch_size, token_length, embedding_dim) with 0-padding.
            return_dict: Whether or not to return a `TokenPoolingOutput` object (with the cluster id to token indices
                         mapping) instead of just the pooled embeddings.
            padding: Whether or not to unbind the padded 3D tensor into a list of 2D tensors. Does nothing if the input
                     is a list of 2D tensors.
            padding_side: The side where the padding was applied in the 3D tensor.
            num_workers: Number of workers for parallel processing. If None, processing is done sequentially.

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

        self._validate_embeddings(embeddings)
        prepared_embeddings = self._prepare_embeddings(embeddings, padding, padding_side)

        # Apply pooling implementation
        pooled_embeddings, cluster_id_to_indices = self._pool_embeddings_impl(
            prepared_embeddings,
            num_workers=num_workers,
        )

        # If the input was a 3D tensor, we need to repad the pooled embeddings for the output to be a 3D
        # tensor as well.
        if isinstance(embeddings, torch.Tensor) and embeddings.dim() == 3:
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


class HierarchicalTokenPooler(BaseTokenPooler):
    """
    Hierarchical token pooling of multi-vector embeddings based on the similarity between token embeddings.

    Example with a list of 2D tensors:

    ```python
    list_embeddings = [torch.rand(10, 768), torch.rand(20, 768)]
    pooler = HierarchicalTokenPooler()
    outputs = pooler.pool_embeddings(list_embeddings, pool_factor=2)
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
    pooler = HierarchicalTokenPooler()
    outputs = pooler.pool_embeddings(list_embeddings, pool_factor=2, padding=True, padding_side="left")
    ```
    """

    def _pool_embeddings_impl(
        self,
        embeddings: List[torch.Tensor],
        pool_factor: int,
        num_workers: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[Dict[int, Tuple[torch.Tensor]]]]:
        """
        Apply hierarchical pooling to each embedding in the list.

        Args:
            embeddings: A list of 2D tensors (token_length, embedding_dim) where each tensor can have its own token
                        length, or a 3D tensor of shape (batch_size, token_length, embedding_dim) with 0-padding.
            pool_factor: An integer factor that determines the maximum number of clusters defined as
                         `max_clusters = max(token_length // pool_factor, 1)`.
            num_workers: The number of workers to use for parallel processing. If not provided, the pooler will use
                         the number of available CPU cores.

        Returns:
            Tuple containing:
            - List of pooled embeddings
            - List of dictionaries mapping cluster IDs to token indices
        """
        with ThreadPoolExecutor(num_workers) as executor:
            # NOTE: We opted for a thread-based pool because most of the heavy lifting is done in C-level libraries
            # (NumPy, Torch, and SciPy) which usually release the GIL.
            results = list(
                executor.map(
                    lambda x: self._pool_single_embedding(x, pool_factor=pool_factor),
                    embeddings,
                )
            )

        # Unpack the results
        pooled_embeddings = [result[0] for result in results]
        cluster_id_to_indices = [result[1] for result in results]

        return pooled_embeddings, cluster_id_to_indices

    def _pool_single_embedding(
        self,
        embedding: torch.Tensor,
        pool_factor: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor]]]:
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

        if pool_factor == 1:
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
        max_clusters = max(token_length // pool_factor, 1)
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


class LambdaTokenPooler(BaseTokenPooler):
    """
    Token pooler that applies a user-defined pooling function to multi-vector embeddings.

    This pooler allows users to define custom pooling methods rather than relying on pre-defined pooling strategies.

    Example:

    ```python
    # Define a custom pooling function that reduces sequence length by half
    def custom_pooling(embedding: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor]]]:
        token_length = embedding.size(0)
        # Resize to half the original length by averaging pairs of tokens
        half_length = token_length // 2 + (token_length % 2)
        pooled_embeddings = torch.zeros((half_length, embedding.size(1)), dtype=embedding.dtype, device=embedding.device)

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

    # Create a LambdaTokenPooler with the custom function
    pooler = LambdaTokenPooler(pool_func=custom_pooling)
    outputs = pooler.pool_embeddings(embeddings)
    ```
    """

    def __init__(
        self,
        pool_func: Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[Dict[int, Tuple[torch.Tensor]]]]],
    ):
        """
        Initialize the LambdaTokenPooler with a custom pooling function.

        Args:
            pool_func: A function that takes a 2D tensor (token_length, embedding_dim) and returns a tuple with:
                      - pooled_embedding: A tensor of shape (num_clusters, embedding_dim)
                      - cluster_id_to_indices (optional): A dictionary mapping cluster ID to token indices
        """
        self.pool_func = pool_func

    def _pool_embeddings_impl(
        self,
        embeddings: List[torch.Tensor],
        num_workers: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[Optional[Dict[int, Tuple[torch.Tensor]]]]]:
        """
        Apply the custom pooling function to each embedding in the list.

        Args:
            embeddings: List of 2D tensors to pool
            num_workers: The number of workers to use for parallel processing. If not provided, the pooler will use
                         the number of available CPU cores.

        Returns:
            Tuple containing:
            - List of pooled embeddings
            - List of dictionaries mapping cluster IDs to token indices
        """
        if num_workers and num_workers > 1:
            with ThreadPoolExecutor(num_workers) as executor:
                # NOTE: We opted for a thread-based pool because most of the heavy lifting is done in C-level libraries
                # (NumPy, Torch, and SciPy) which usually release the GIL.
                results = list(executor.map(self.pool_func, embeddings))
        else:
            # Process sequentially
            results = [self.pool_func(emb) for emb in embeddings]

        # Unpack the results
        pooled_embeddings = [result[0] for result in results]
        cluster_id_to_indices = [result[1] for result in results]

        return pooled_embeddings, cluster_id_to_indices
