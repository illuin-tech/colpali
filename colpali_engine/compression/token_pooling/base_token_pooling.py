from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, cast

import torch

from colpali_engine.utils.torch_utils import unbind_padded_multivector_embeddings


@dataclass
class TokenPoolingOutput:
    """
    Token pooling outputs:
    - pooled_embeddings: A list of 2D tensors (token_length, embedding_dim) where each tensor can have its own
                         token_length, or a 3D tensor of shape (batch_size, token_length, embedding_dim) with
                         optional padding.
    - cluster_id_to_indices (optional): A list of dictionaries. The i-th dictionary maps the cluster id to token indices
                                        for the i-th embedding in `pooled_embeddings`.
    """

    pooled_embeddings: Union[List[torch.Tensor], torch.Tensor]
    cluster_id_to_indices: Optional[Dict[int, Tuple[torch.Tensor]]] = None


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
    ) -> Tuple[
        List[torch.Tensor],
        Optional[List[Dict[int, Tuple[torch.Tensor]]]],
    ]:
        """
        Implementation of pooling logic for a list of 2D embeddings.

        Args:
            embeddings: A list of 2D tensors (token_length, embedding_dim)
            num_workers: Number of workers for parallel processing

        Returns:
            Tuple containing:
            - List of pooled embeddings
            - (Optional) List of dictionaries mapping cluster IDs to token indices
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
        **pool_kwargs,
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
            **pool_kwargs,
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
