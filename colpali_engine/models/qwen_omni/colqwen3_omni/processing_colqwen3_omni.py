from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

from .processing_bidirlm_omni import BidirLMOmniProcessor


class ColQwen3OmniProcessor(BaseVisualRetrieverProcessor, BidirLMOmniProcessor):
    """
    Processor for ColQwen3-Omni / BidirLM-Omni checkpoints.
    """

    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    audio_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>Describe the sound.<|im_end|><|endoftext|>"
    )
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        max_image_size: Optional[int] = None,
    ):
        super().__init__(
            image_processor=image_processor,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_image_size=max_image_size,
        )
        self.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        *args,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        max_num_visual_tokens = kwargs.pop("max_num_visual_tokens", None)
        instance = super().from_pretrained(
            *args,
            device_map=device_map,
            **kwargs,
        )

        if max_num_visual_tokens is not None:
            patch_size = getattr(instance.image_processor, "patch_size", None)
            merge_size = getattr(instance.image_processor, "merge_size", None)
            if patch_size is None or merge_size is None:
                raise ValueError("BidirLM-Omni image processor is missing `patch_size` or `merge_size`.")
            tile = patch_size * merge_size
            instance.image_processor.max_pixels = max_num_visual_tokens * tile * tile
            instance.image_processor.size["longest_edge"] = instance.image_processor.max_pixels

        return instance

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )

        offsets = batch_doc["image_grid_thw"].prod(dim=1)
        pixel_values = list(torch.split(batch_doc["pixel_values"], offsets.tolist()))
        batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(pixel_values, batch_first=True)

        return batch_doc

    def process_audios(self, audios) -> Union[BatchFeature, BatchEncoding]:
        return self(
            text=[self.audio_prompt_prefix] * len(audios),
            audio=audios,
            padding="longest",
            return_tensors="pt",
        )

    def process_videos(self, videos) -> Union[BatchFeature, BatchEncoding]:
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video},
                        {"type": "text", "text": "Describe the video."},
                    ],
                }
            ]
            for video in videos
        ]
        text = [
            self.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            for conversation in conversations
        ]

        batch_doc = self(
            text=text,
            videos=videos,
            padding="longest",
            return_tensors="pt",
        )

        offsets = batch_doc["video_grid_thw"].prod(dim=1)
        pixel_values_videos = list(torch.split(batch_doc["pixel_values_videos"], offsets.tolist()))
        batch_doc["pixel_values_videos"] = torch.nn.utils.rnn.pad_sequence(pixel_values_videos, batch_first=True)

        return batch_doc

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
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
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        patch_size = self.image_processor.patch_size

        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=patch_size * self.image_processor.merge_size,
            min_pixels=self.image_processor.size["shortest_edge"],
            max_pixels=self.image_processor.size["longest_edge"],
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id
