from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import BatchFeature, PaliGemmaProcessor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColPaliProcessor(BaseVisualRetrieverProcessor, PaliGemmaProcessor):
    """
    Processor for ColPali.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock_image = Image.new("RGB", (16, 16), color="black")

    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature:
        """
        Process images for ColPali.
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
        Process queries for ColPali.
        """
        # NOTE: The image is required for calling PaligemmaProcessor, so we create a mock image here.

        suffix = suffix or "<pad>" * 10
        texts_query: List[str] = []

        for query in queries:
            query = f"Question: {query}"
            query += suffix  # add suffix (pad tokens)
            texts_query.append(query)

        batch_query = self(
            images=[self.mock_image] * len(texts_query),
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            max_length=max_length + self.image_seq_length,
        )

        del batch_query["pixel_values"]

        batch_query["input_ids"] = batch_query["input_ids"][..., self.image_seq_length :]
        batch_query["attention_mask"] = batch_query["attention_mask"][..., self.image_seq_length :]

        return batch_query

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        batch_size: int = 128,
        device: Union[Optional[str], torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """

        return self.score_multi_vector(qs, ps, batch_size=batch_size, device=device)
