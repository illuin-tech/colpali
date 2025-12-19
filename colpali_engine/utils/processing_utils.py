import importlib
import logging
import math
from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature

try:
    from fast_plaid import search
except ImportError:
    logging.info(
        "FastPlaid is not installed.If you want to use it:Instal with `pip install --no-deps fast-plaid fastkmeans`"
    )

from colpali_engine.utils.torch_utils import get_torch_device


class BaseVisualRetrieverProcessor(ABC):
    """
    Base class for visual retriever processors.
    """

    query_prefix: ClassVar[str] = ""  # Default prefix for queries. Override in subclasses if needed.

    @abstractmethod
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process a list of images into a format suitable for the model.
        Args:
            images (List[Image.Image]): List of images to process.
        Returns:
            Union[BatchFeature, BatchEncoding]: Processed images.
        """
        pass

    @abstractmethod
    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process a list of texts into a format suitable for the model.

        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
        """
        pass

    def process_queries(
        self,
        texts: Optional[List[str]] = None,
        queries: Optional[List[str]] = None,
        max_length: int = 50,
        contexts: Optional[List[str]] = None,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process a list of queries into a format suitable for the model.

        Args:
            texts: List of input texts.
            [DEPRECATED] max_length: Maximum length of the text.
            suffix: Suffix to append to each text. If None, the default query augmentation token is used.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.

        NOTE: This function will be deprecated. Use `process_texts` instead.
        It is kept to maintain back-compatibility with vidore evaluator.
        """

        if texts and queries:
            raise ValueError("Only one of 'texts' or 'queries' should be provided.")
        if queries is not None:
            texts = queries
        elif texts is None:
            raise ValueError("No texts or queries provided.")

        if suffix is None:
            suffix = self.query_augmentation_token * 10

        # Add the query prefix and suffix to each text
        texts = [self.query_prefix + text + suffix for text in texts]

        return self.process_texts(texts=texts)

    @abstractmethod
    def score(
        self,
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def score_single_vector(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        device = device or get_torch_device("auto")

        if isinstance(qs, list) and isinstance(ps, list):
            if len(qs) == 0:
                raise ValueError("No queries provided")
            if len(ps) == 0:
                raise ValueError("No passages provided")

            qs = torch.stack(qs).to(device)
            ps = torch.stack(ps).to(device)
        else:
            qs = qs.to(device)
            ps = ps.to(device)

        scores = torch.einsum("bd,cd->bc", qs, ps)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def score_multi_vector(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            device (`Union[str, torch.device]`, *optional*): Device to use for computation. If not
                provided, uses `get_torch_device("auto")`.

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                device
            )
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to(device)
                scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def get_topk_plaid(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        plaid_index: "search.FastPlaid",
        k: int = 10,
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Experimental: Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings endoded in a plaid index. For ColPali, a passage is the
        image of a document page.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                device
            )
            # Use the plaid index to get the top-k scores
            scores_batch = plaid_index.search(
                queries_embeddings=qs_batch.to(torch.float32),
                top_k=k,
            )
            scores_list.append(scores_batch)

        return scores_list

    @staticmethod
    def create_plaid_index(
        ps: Union[torch.Tensor, List[torch.Tensor]],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Experimental: Create a FastPlaid index from the given passage embeddings.
        Args:
            ps (`Union[torch.Tensor, List[torch.Tensor]]`): Passage embeddings. Should be a list of tensors,
                where each tensor is of shape (sequence_length_i, embedding_dim).
            device (`Optional[Union[str, torch.device]]`, *optional*): Device to use for computation. If not
                provided, uses `get_torch_device("auto")`.
        """
        # assert fast_plaid is installed
        if not importlib.util.find_spec("fast_plaid"):
            raise ImportError("FastPlaid is not installed. Please install it with `pip install fast-plaid`.")

        fast_plaid_index = search.FastPlaid(index="index")
        # torch.nn.utils.rnn.pad_sequence(ds, batch_first=True, padding_value=0).to(device)
        device = device or get_torch_device("auto")
        fast_plaid_index.create(documents_embeddings=[d.to(device).to(torch.float32) for d in ps])
        return fast_plaid_index

    @abstractmethod
    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        *args,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an
        image of size (height, width) with the given patch size.
        """
        pass


class Idefics3SplitImageInterpretabilityMixin:
    """
    Mixin class providing interpretability support for Idefics3-style image splitting processors.

    This mixin adds methods for:
    - Getting image token masks (full and local-only)
    - Calculating patch grid dimensions
    - Rearranging embeddings from sub-patch order to spatial order
    - Computing similarity maps with correct spatial correspondence

    This is designed for processors that use Idefics3-style image splitting where:
    1. Images are resized to fit within a longest_edge constraint
    2. Images are split into sub-patches (e.g., 512x512 patches)
    3. Each sub-patch becomes image_seq_len tokens (e.g., 64 tokens in an 8x8 grid)
    4. A global patch is added as the last image_seq_len tokens

    Both ColIdefics3Processor and ColModernVBertProcessor use this pattern.
    """

    # These attributes must be provided by the implementing class
    image_token: str  # e.g., "<image>"
    image_seq_len: int  # e.g., 64
    tokenizer: any  # The tokenizer instance
    image_processor: any  # The image processor instance

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """
        Get a tensor mask that identifies the image tokens in the batch.

        Args:
            batch_images: BatchFeature containing processed images with input_ids.

        Returns:
            A boolean tensor of the same shape as input_ids, where True indicates
            an image token position.
        """
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        return batch_images.input_ids == image_token_id

    def get_local_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """
        Get a tensor mask that identifies only the LOCAL image tokens in the batch,
        excluding the global patch tokens.

        In Idefics3 with image splitting, images are split into multiple sub-patches
        plus one global patch. The global patch tokens are the last image_seq_len
        image tokens for each image. For interpretability purposes, we typically want
        to exclude the global patch since it doesn't have spatial correspondence.

        Args:
            batch_images: BatchFeature containing processed images with input_ids.

        Returns:
            A boolean tensor of the same shape as input_ids, where True indicates
            a LOCAL image token position (excluding global patch).
        """
        # Get the full image mask first
        full_mask = self.get_image_mask(batch_images)
        local_mask = full_mask.clone()

        # For each batch item, exclude the last image_seq_len image tokens (global patch)
        batch_size = batch_images.input_ids.shape[0]

        for batch_idx in range(batch_size):
            # Find all image token positions in this batch item
            image_positions = full_mask[batch_idx].nonzero(as_tuple=True)[0]

            if len(image_positions) > self.image_seq_len:
                # Exclude the last image_seq_len tokens (global patch)
                global_patch_start = len(image_positions) - self.image_seq_len
                global_patch_indices = image_positions[global_patch_start:]

                # Set these positions to False in the local mask
                for idx in global_patch_indices:
                    local_mask[batch_idx, idx] = False

        return local_mask

    def _calculate_resized_dimensions(
        self,
        image_size: Tuple[int, int],
        longest_edge: Optional[int],
    ) -> Tuple[int, int]:
        """
        Calculate the resized dimensions for an image based on the longest_edge constraint.

        This mirrors the Idefics3ImageProcessor logic for resizing images.

        Args:
            image_size: Tuple of (height, width) in pixels.
            longest_edge: Maximum size for the longest edge. If None, no resizing is applied.

        Returns:
            Tuple of (height_new, width_new) representing the resized dimensions.
        """
        height, width = image_size

        # Handle edge case where resizing is disabled (research use case)
        if longest_edge is None:
            return height, width

        # Resize the image so the longest edge equals longest_edge
        aspect_ratio = width / height

        if width >= height:
            # Width is the longest edge
            width_new = longest_edge
            height_new = int(width_new / aspect_ratio)
            # Ensure height is even (as per Idefics3 implementation)
            if height_new % 2 != 0:
                height_new += 1
        else:
            # Height is the longest edge
            height_new = longest_edge
            width_new = int(height_new * aspect_ratio)
            # Ensure width is even (as per Idefics3 implementation)
            if width_new % 2 != 0:
                width_new += 1

        # Ensure minimum size of 1
        height_new = max(height_new, 1)
        width_new = max(width_new, 1)

        return height_new, width_new

    def rearrange_image_embeddings(
        self,
        image_embeddings: torch.Tensor,
        image_mask: torch.Tensor,
        n_patches: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Rearrange image embeddings from sub-patch order to spatial order.

        In Idefics3 with image splitting, tokens are arranged sub-patch by sub-patch:
        - Tokens 0-63: First 512x512 sub-patch (8x8 tokens)
        - Tokens 64-127: Second 512x512 sub-patch (8x8 tokens)
        - etc.

        This method rearranges them into a proper 2D spatial grid where tokens
        are organized by their actual spatial position in the image.

        Args:
            image_embeddings: tensor of shape (sequence_length, dim) for a single image
            image_mask: boolean tensor of shape (sequence_length,) indicating image tokens
            n_patches: tuple of (n_patches_x, n_patches_y) - total token grid dimensions

        Returns:
            tensor of shape (n_patches_x, n_patches_y, dim) with spatially correct ordering
        """
        # Extract only the image token embeddings
        masked_embeddings = image_embeddings[image_mask]  # (n_patches_x * n_patches_y, dim)

        n_patches_x, n_patches_y = n_patches
        dim = masked_embeddings.shape[-1]

        # Calculate sub-patch grid dimensions
        tokens_per_subpatch_side = int(math.sqrt(self.image_seq_len))
        n_subpatches_x = n_patches_x // tokens_per_subpatch_side
        n_subpatches_y = n_patches_y // tokens_per_subpatch_side

        # Reshape from flat sub-patch order to sub-patch grid
        # Current order: (n_subpatches_y * n_subpatches_x * tokens_per_side * tokens_per_side, dim)
        # Reshape to: (n_subpatches_y, n_subpatches_x, tokens_per_side, tokens_per_side, dim)
        reshaped = masked_embeddings.reshape(
            n_subpatches_y,
            n_subpatches_x,
            tokens_per_subpatch_side,
            tokens_per_subpatch_side,
            dim,
        )

        # Permute to interleave sub-patch rows and columns
        # From: (n_subpatches_y, n_subpatches_x, tokens_per_side, tokens_per_side, dim)
        # To: (n_subpatches_y, tokens_per_side, n_subpatches_x, tokens_per_side, dim)
        permuted = reshaped.permute(0, 2, 1, 3, 4)

        # Final reshape to (n_patches_y, n_patches_x, dim)
        # Note: This gives (height, width, dim) ordering
        spatial_grid = permuted.reshape(n_patches_y, n_patches_x, dim)

        # Transpose to get (n_patches_x, n_patches_y, dim) to match expected format
        # This gives (width, height, dim) ordering which matches the similarity map convention
        spatial_grid = spatial_grid.permute(1, 0, 2)

        return spatial_grid

    def get_similarity_maps_from_embeddings(
        self,
        image_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
        n_patches: Union[Tuple[int, int], List[Tuple[int, int]]],
        image_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Get similarity maps with correct spatial ordering for Idefics3-style image splitting.

        This method correctly handles the sub-patch token ordering used by Idefics3 processors,
        where tokens are arranged sub-patch by sub-patch rather than in row-major order across
        the entire image.

        Args:
            image_embeddings: tensor of shape (batch_size, image_tokens, dim)
            query_embeddings: tensor of shape (batch_size, query_tokens, dim)
            n_patches: number of patches per dimension (n_patches_x, n_patches_y).
                If a single tuple, it's broadcasted to all batch items.
            image_mask: tensor of shape (batch_size, image_tokens) indicating LOCAL image tokens
                (use get_local_image_mask to exclude global patch)

        Returns:
            List of tensors, each of shape (query_tokens, n_patches_x, n_patches_y)
        """
        if isinstance(n_patches, tuple):
            n_patches = [n_patches] * image_embeddings.size(0)

        similarity_maps: List[torch.Tensor] = []

        for idx in range(image_embeddings.size(0)):
            # Sanity check
            if image_mask[idx].sum() != n_patches[idx][0] * n_patches[idx][1]:
                raise ValueError(
                    f"The number of patches ({n_patches[idx][0]} x {n_patches[idx][1]} = "
                    f"{n_patches[idx][0] * n_patches[idx][1]}) "
                    f"does not match the number of non-padded image tokens ({image_mask[idx].sum()}). "
                    f"Hint: Use get_local_image_mask() instead of get_image_mask() to exclude the global patch."
                )

            # Rearrange image embeddings to correct spatial order
            image_embedding_grid = self.rearrange_image_embeddings(
                image_embeddings[idx],
                image_mask[idx],
                n_patches[idx],
            )  # (n_patches_x, n_patches_y, dim)

            # Compute similarity: einsum("nk,ijk->nij", query, image_grid)
            # query: (query_tokens, dim)
            # image_grid: (n_patches_x, n_patches_y, dim)
            # result: (query_tokens, n_patches_x, n_patches_y)
            similarity_map = torch.einsum("nk,ijk->nij", query_embeddings[idx], image_embedding_grid)

            similarity_maps.append(similarity_map)

        return similarity_maps
