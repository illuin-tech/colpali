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
        prefix = "Question: "

        if suffix is None:
            suffix = "<pad>" * 10
        texts_query: List[str] = []

        for query in queries:
            query = self.tokenizer.bos_token + prefix + query
            query += suffix  # add suffix (pad tokens)

            # NOTE: Make input ISO to PaliGemma's processor
            query += "\n"

            texts_query.append(query)

        batch_query = self.tokenizer(
            texts_query,
            text_pair=None,
            return_token_type_ids=False,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
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
