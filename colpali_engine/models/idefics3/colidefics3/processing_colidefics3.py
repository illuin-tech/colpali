import math
from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Idefics3Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColIdefics3Processor(BaseVisualRetrieverProcessor, Idefics3Processor):
    """
    Processor for ColIdefics3.
    """

    query_augmentation_token: ClassVar[str] = "<end_of_utterance>"
    image_token: ClassVar[str] = "<image>"
    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
    )

    def __init__(self, *args, image_seq_len=64, **kwargs):
        super().__init__(*args, image_seq_len=image_seq_len, **kwargs)
        self.tokenizer.padding_side = "left"

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for ColIdefics3.

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
        Process texts for ColIdefics3.

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
        patch_size: int,
        *args,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        This method mirrors the Idefics3 image processing logic:
        1. Resize the image so the longest edge equals the processor's longest_edge setting
        2. Calculate the number of patches in each direction using ceiling division

        Args:
            image_size: Tuple of (height, width) in pixels.
            patch_size: The size of each square patch in pixels.

        Returns:
            Tuple of (n_patches_x, n_patches_y) representing the number of patches
            along the width and height dimensions respectively.
        """
        height, width = image_size

        # Get the longest_edge from the image processor's size configuration
        # This is the maximum size for the longest edge after resizing
        longest_edge = self.image_processor.size.get("longest_edge", 4 * patch_size)

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

        # Step 2: Calculate the number of patches in each direction
        # This mirrors the split_image logic from Idefics3ImageProcessor
        n_patches_y = math.ceil(height_new / patch_size)
        n_patches_x = math.ceil(width_new / patch_size)

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
