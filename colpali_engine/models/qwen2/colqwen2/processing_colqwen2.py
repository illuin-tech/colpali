from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import BatchFeature
from transformers.models.qwen2_vl import Qwen2VLProcessor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

from qwen_vl_utils import smart_resize

class ColQwen2Processor(BaseVisualRetrieverProcessor, Qwen2VLProcessor):
    """
    Processor for ColQwen2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "right"
        self.min_pixels =  4 * 28 * 28
        self.max_pixels =  512 * 28 * 28
        self.factor = 28

    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature:
        """
        Process images for ColPali.
        """
        texts_doc = (["<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n"]
                     * len(images))

        def resize_and_convert(image):
            image_size = image.size
            resized_height, resized_width = smart_resize(image_size[1], image_size[0], factor=self.factor, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            print(f"Resizing image from {image_size} to {(resized_height, resized_width)}")
            return image.convert("RGB").resize((resized_width, resized_height))

        images = [resize_and_convert(image) for image in images]

        # images = [image.convert("RGB").resize((448, 448)) for image in images]

        batch_doc = self(
            text=texts_doc,
            images=images,
            padding="longest",
            return_tensors="pt"
        )

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
