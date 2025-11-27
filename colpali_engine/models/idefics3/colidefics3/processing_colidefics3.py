import math
from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Idefics3Processor
from transformers.models.idefics3.image_processing_idefics3 import (
    _resize_output_size_rescale_to_max_len,
    _resize_output_size_scale_below_upper_bound,
)

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColIdefics3Processor(BaseVisualRetrieverProcessor, Idefics3Processor):
    """
    Processor for ColIdefics3.
    """

    query_augmentation_token: ClassVar[str] = "<end_of_utterance>"
    image_token: ClassVar[str] = "<image>"
    visual_prompt_prefix: ClassVar[str] = "<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"

    def __init__(self, *args, image_seq_len=64, **kwargs):
        super().__init__(*args, image_seq_len=image_seq_len, **kwargs)

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
        *args,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Calculate the number of patches for the given image size and patch size.

        For Idefics3, we replicate the complete image processing pipeline to determine
        the final resized dimensions before calculating patches. The total number of patches
        is the grid size multiplied by the patch subdivision (sqrt of image_seq_len).

        If the processor's do_resize is disabled, Steps 1-2 (rescaling) are skipped and
        only grid alignment is applied to the original image dimensions.

        Args:
            image_size: The size of the original image as (width, height)

        Returns:
            Tuple[int, int]: Number of patches in (x, y) dimensions
        """
        width, height = image_size

        # Get processor parameters
        max_image_size_edge = self.image_processor.max_image_size["longest_edge"]  # Default: 364
        size_longest_edge = self.image_processor.size["longest_edge"]  # Default: 4 * 364 = 1456

        # Check if resizing is enabled
        do_resize = self.image_processor.do_resize

        # Apply Steps 1-2 only if resizing is enabled
        if do_resize:
            # Step 1: Rescale to max length (preserving aspect ratio)
            height, width = _resize_output_size_rescale_to_max_len(height, width, min_len=1, max_len=size_longest_edge)

            # Step 2: Scale below upper bound (4096)
            height, width = _resize_output_size_scale_below_upper_bound(height, width, max_len=4096)

        # Step 3: Upscale to multiples of max_image_size (patch grid alignment)
        if width >= height:
            resized_width = math.ceil(width / max_image_size_edge) * max_image_size_edge
            resized_height = math.ceil(height / max_image_size_edge) * max_image_size_edge
        elif height > width:
            resized_height = math.ceil(height / max_image_size_edge) * max_image_size_edge
            resized_width = math.ceil(width / max_image_size_edge) * max_image_size_edge

        # Step 4: Calculate number of grid cells
        n_grid_x = resized_width // max_image_size_edge
        n_grid_y = resized_height // max_image_size_edge

        # Step 5: Calculate actual number of patches
        # Each grid cell contains image_seq_len tokens arranged as a square grid
        patches_per_grid_side = int(math.sqrt(self.image_seq_len))
        n_patches_x = n_grid_x * patches_per_grid_side
        n_patches_y = n_grid_y * patches_per_grid_side

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """
        Get a mask for image tokens, excluding the global image representation.

        In Idefics3, images are split into patches, and a global image representation
        is added at the end. This method returns a mask that includes only the patch-level
        image tokens, excluding the global image tokens.

        The global image section is always the last group of `image_seq_len` consecutive
        <image> tokens in each sequence.

        Args:
            batch_images: BatchFeature containing input_ids

        Returns:
            torch.Tensor: Boolean mask where True indicates patch-level image tokens
        """
        # Start with a mask of all image tokens
        image_mask = batch_images.input_ids == self.image_token_id

        batch_size = image_mask.shape[0]

        for batch_idx in range(batch_size):
            # Find all positions where image tokens occur
            image_positions = torch.where(image_mask[batch_idx])[0]

            if len(image_positions) == 0:
                continue

            # The global image is the last image_seq_len consecutive image tokens
            # We need to identify and exclude them
            if len(image_positions) >= self.image_seq_len:
                # Get the last image_seq_len image token positions
                global_image_positions = image_positions[-self.image_seq_len :]

                # Verify they are consecutive (they should be)
                if torch.all(global_image_positions[1:] - global_image_positions[:-1] == 1):
                    # Set the global image tokens to False in the mask
                    image_mask[batch_idx, global_image_positions] = False

        return image_mask
