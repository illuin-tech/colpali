from __future__ import annotations

from typing import List, Optional, cast

from PIL import Image
from transformers import BatchEncoding, Idefics2Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColIdefics2Processor(BaseVisualRetrieverProcessor):
    """
    Processor for ColIdefics2.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "HuggingFaceM4/idefics2-8b",
    ):
        super().__init__()
        self.processor = cast(Idefics2Processor, Idefics2Processor.from_pretrained(pretrained_model_name_or_path))

    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchEncoding:
        """
        Process images for ColIdefics2, with an efficient tweak around the Idefics2 processor.
        """
        texts_doc = []
        images = [image.convert("RGB") for image in images]

        for _ in images:
            messages_doc = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image."},
                        {"type": "image"},
                    ],
                },
            ]

            text_doc = self.processor.apply_chat_template(messages_doc, add_generation_prompt=False)
            texts_doc.append(text_doc.strip())

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
        Process queries for ColIdefics2, with an efficient tweak around the Idefics2 processor.
        """
        suffix = suffix or "<end_of_utterance>" * 5
        texts_query = []
        for query in queries:
            messages_query = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Question: {query}" + suffix,
                        },
                    ],
                },
            ]
            text_query = self.processor.apply_chat_template(messages_query, add_generation_prompt=False).strip()
            texts_query.append(text_query)

        batch_query = self.processor(
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
        )
        return batch_query
