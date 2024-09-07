from __future__ import annotations

from typing import List, Optional, cast

from PIL import Image
from transformers import BatchEncoding, BatchFeature, LlamaTokenizerFast, PaliGemmaProcessor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColPaliProcessor(BaseVisualRetrieverProcessor):
    """
    Processor for ColPali.
    """

    def __init__(
        self,
        vlm_backbone_model_name_or_path: str = "google/paligemma-3b-mix-448",
        hf_token: Optional[str] = None,
    ):
        super().__init__()

        # TODO: After ColPali integration in transformers where ColPaliProcessor will copy
        # PaligemmaProcessor, remove `self.processor` and use `self` directly.
        self.processor = cast(
            PaliGemmaProcessor,
            PaliGemmaProcessor.from_pretrained(
                vlm_backbone_model_name_or_path,
                token=hf_token,
            ),
        )
        self.tokenizer = cast(LlamaTokenizerFast, self.processor.tokenizer)  # type: ignore

    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature:
        """
        Process images for ColPali.
        """
        texts_doc = ["Describe the image."] * len(images)
        images = [image.convert("RGB") for image in images]

        batch_doc = self.processor(
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
    ) -> BatchEncoding:
        """
        Process queries for ColPali.

        NOTE: This method uses the PaliGemma tokenizer instead of the processor because
        the processor cannot handle text without images.
        """
        suffix = suffix or "<pad>" * 10
        texts_query: List[str] = []

        for query in queries:
            query = f"Question: {query}"
            query += suffix  # add suffix (pad tokens)
            texts_query.append(query)

        batch_query = self.tokenizer(
            texts_query,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
        )
        return batch_query
