from __future__ import annotations

from typing import List, Optional, cast

from PIL import Image
from transformers import BatchFeature, PaliGemmaProcessor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColPaliProcessor(BaseVisualRetrieverProcessor, PaliGemmaProcessor):
    def __init__(self):
        BaseVisualRetrieverProcessor.__init__(self)
        PaliGemmaProcessor.__init__(self)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cast(cls, PaliGemmaProcessor.from_pretrained(*args, **kwargs))

    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature:
        """
        Process images for ColPali, with an efficient tweak around the PaliGemmma processor.
        """
        texts_doc = ["Describe the image."] * len(images)
        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=texts_doc,
            images=images,
            return_tensors="pt",
            padding="longest",
        )

        return batch_doc

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for ColPali, with an efficient tweak around the PaliGemmma processor.
        """
        # NOTE: The image is required for calling PaligemmaProcessor, so we create a mock image here.
        mock_image = Image.new("RGB", (448, 448), (255, 255, 255)).convert("RGB")

        suffix = suffix or "<pad>" * 10
        texts_query: List[str] = []

        for query in queries:
            query = f"Question: {query}"
            query += suffix  # add suffix (pad tokens)
            texts_query.append(query)

        batch_query = self(
            images=[mock_image] * len(texts_query),
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            max_length=max_length + self.image_seq_length,
        )
        del batch_query["pixel_values"]

        # TODO: call the tokenizer instead?

        batch_query["input_ids"] = batch_query["input_ids"][..., self.image_seq_length :]
        batch_query["attention_mask"] = batch_query["attention_mask"][..., self.image_seq_length :]

        return batch_query
