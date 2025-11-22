import math
from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Idefics3Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColModernVBertProcessor(BaseVisualRetrieverProcessor, Idefics3Processor):
    """
    Processor for ColIdefics3.
    """

    query_augmentation_token: ClassVar[str] = "<end_of_utterance>"
    image_token: ClassVar[str] = "<image>"
    visual_prompt_prefix: ClassVar[str] = (
        "<|begin_of_text|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
    )

    def __init__(self, *args, image_seq_len=64, **kwargs):
        super().__init__(*args, image_seq_len=image_seq_len, **kwargs)
        self.tokenizer.padding_side = "left"

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for ColModernVBert.

        Args:
            images: List of PIL images.
        """
        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )
        return batch_doc

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for ColModernVBert.

        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
        """
        return self(
            text=texts,
            return_tensors="pt",
            padding="longest",
        )

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        This method mirrors the Idefics3 image processing logic with image splitting:
        1. Resize the image so the longest edge equals the processor's longest_edge setting
        2. Split into sub-patches of max_image_size (512x512)
        3. Each sub-patch becomes image_seq_len tokens (8x8 grid)

        Note: The patch_size parameter is kept for API compatibility but is not used in the
        calculation. The actual patch dimensions are determined by the image splitting logic
        and image_seq_len.

        Args:
            image_size: Tuple of (height, width) in pixels.
            patch_size: The size of each square patch in pixels (unused, kept for API compatibility).

        Returns:
            Tuple of (n_patches_x, n_patches_y) representing the number of token patches
            along the width and height dimensions respectively (excluding global patch).
        """
        height, width = image_size

        # Get the longest_edge from the image processor's size configuration
        # This is the maximum size for the longest edge after resizing
        longest_edge = self.image_processor.size.get("longest_edge", 2048)

        # Handle edge case where resizing is disabled (research use case)
        # When longest_edge is None, use original dimensions without resizing
        if longest_edge is None:
            height_new, width_new = height, width
        else:
            # Step 1: Resize the image so the longest edge equals longest_edge
            # This mirrors _resize_output_size_rescale_to_max_len from Idefics3ImageProcessor
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

        # Step 2: Calculate number of sub-patches (512x512 patches)
        # This mirrors the split_image logic from Idefics3ImageProcessor
        max_image_size = self.image_processor.max_image_size.get("longest_edge", 512)
        n_subpatches_x = math.ceil(width_new / max_image_size)
        n_subpatches_y = math.ceil(height_new / max_image_size)

        # Step 3: Calculate token grid dimensions
        # Each sub-patch becomes image_seq_len tokens (typically 64 = 8x8 grid)
        tokens_per_subpatch_side = int(math.sqrt(self.image_seq_len))
        n_patches_x = n_subpatches_x * tokens_per_subpatch_side
        n_patches_y = n_subpatches_y * tokens_per_subpatch_side

        return n_patches_x, n_patches_y

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
        masked_embeddings = image_embeddings[
            image_mask
        ]  # (n_patches_x * n_patches_y, dim)

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
            similarity_map = torch.einsum(
                "nk,ijk->nij", query_embeddings[idx], image_embedding_grid
            )

            similarity_maps.append(similarity_map)

        return similarity_maps
