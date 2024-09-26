import math
from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import BatchFeature
from transformers.models.qwen2_vl import Qwen2VLProcessor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColQwen2Processor(BaseVisualRetrieverProcessor, Qwen2VLProcessor):
    """
    Processor for ColQwen2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"
        self.min_pixels =  4 * 28 * 28
        self.max_pixels =  768 * 28 * 28
        self.factor = 28
        self.max_ratio = 200

    @staticmethod
    def round_by_factor(number: float, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    @staticmethod
    def ceil_by_factor(number: float, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    @staticmethod
    def floor_by_factor(number: float, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor


    def smart_resize(self, height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """
        if max(height, width) / min(height, width) > self.max_ratio:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {self.max_ratio}, "
                f"got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature:
        """
        Process images for ColPali.
        """
        texts_doc = (["<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n"]
                     * len(images))

        def resize_and_convert(image: Image.Image) -> Image.Image:
            image_size = image.size
            resized_height, resized_width = self.smart_resize(image_size[1],
                                                         image_size[0],
                                                         factor=self.factor,
                                                         min_pixels=self.min_pixels,
                                                         max_pixels=self.max_pixels)
            # print(f"Resizing image from {image_size} to {(resized_height, resized_width)}")
            return image.convert("RGB").resize((resized_width, resized_height))

        images = [resize_and_convert(image) for image in images]


        batch_doc = self(
            text=texts_doc,
            images=images,
            padding="longest",
            return_tensors="pt"
        )


        # The following code is a hack to make sure the scatter in DDP is done correctly when training on multiple GPUs
        offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]
        # separate pixel_values for each image
        pixel_values = torch.split(batch_doc["pixel_values"], offsets.tolist())
        # pad pixel_values to the same length to be able to make it into a tensor
        max_length = max([len(pv) for pv in pixel_values])
        pixel_values = [torch.cat([pv,
                                   torch.zeros((max_length - len(pv), pv.shape[1]),
                                               dtype=pv.dtype, device=pv.device)]) for pv in pixel_values]
        batch_doc["pixel_values"] = torch.stack(pixel_values)


        return batch_doc

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for ColPali.
        """
        if suffix is None:
            suffix = "<pad>" * 10
        texts_query: List[str] = []

        for query in queries:
            query = f"Query: {query}"
            query += suffix  # add suffix (pad tokens)
            texts_query.append(query)

        batch_query = self(
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            # max_length=max_length + self.image_seq_length,
        )

        return batch_query

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
