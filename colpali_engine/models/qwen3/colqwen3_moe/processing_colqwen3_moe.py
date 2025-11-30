from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature

from colpali_engine.models.qwen3.colqwen3.processing_colqwen3 import ColQwen3Processor


class ColQwen3MoEProcessor(ColQwen3Processor):
    """
    Processor for the ColQwen3-MoE model variant. The MoE backbone shares the same vision and text
    preprocessing pipeline as the dense Qwen3-VL models, but exposing a dedicated class keeps the API
    symmetric with the available ColPali wrappers.
    """

    moe_variant: ClassVar[bool] = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        *args,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        return super().from_pretrained(*args, device_map=device_map, **kwargs)

    def process_images(self, images: List[Image.Image]) -> Union[BatchFeature, BatchEncoding]:
        return super().process_images(images)

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        return super().process_texts(texts)

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return super().score(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        return super().get_n_patches(image_size, spatial_merge_size)

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return super().get_image_mask(batch_images)
