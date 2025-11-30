from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen3_vl import Qwen3VLProcessor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColQwen3Processor(BaseVisualRetrieverProcessor, Qwen3VLProcessor):
    """
    Processor for ColQwen3.

    Args:
        max_num_visual_tokens: Maximum number of visual tokens allowed during preprocessing.
        *args: Variable positional arguments forwarded to :class:`~transformers.Qwen3VLProcessor`.
        **kwargs: Keyword arguments forwarded to :class:`~transformers.Qwen3VLProcessor`.
    """

    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        *args,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        instance = super().from_pretrained(
            *args,
            device_map=device_map,
            **kwargs,
        )

        if "max_num_visual_tokens" in kwargs:
            patch_size = getattr(instance.image_processor, "patch_size", 28)
            instance.image_processor.max_pixels = kwargs["max_num_visual_tokens"] * patch_size * patch_size
            instance.image_processor.size["longest_edge"] = instance.image_processor.max_pixels

        return instance

    def process_images(self, images: List[Image.Image]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process a batch of PIL images for ColQwen3.
        """

        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )

        if batch_doc["pixel_values"].numel() == 0:
            return batch_doc

        offsets = batch_doc["image_grid_thw"].prod(dim=1)
        pixel_values = list(
            torch.split(batch_doc["pixel_values"], offsets.tolist())
        )  # [(num_patches_img_0, patch_dim), ..., (num_patches_img_n, patch_dim)]

        batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(pixel_values, batch_first=True)

        return batch_doc

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process a batch of raw texts for ColQwen3.
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
        Compute the MaxSim score (ColBERT-like) for query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Compute the number of spatial patches for an image of ``image_size``.
        """
        patch_size = self.image_processor.patch_size
        merge_size = getattr(self.image_processor, "merge_size", 1)

        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=patch_size * merge_size,
            min_pixels=self.image_processor.size["shortest_edge"],
            max_pixels=self.image_processor.size["longest_edge"],
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """
        Return a boolean tensor identifying image tokens inside ``batch_images``.
        """
        return batch_images.input_ids == self.image_token_id

