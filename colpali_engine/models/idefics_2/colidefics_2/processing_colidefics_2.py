import warnings
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, Idefics2Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColIdefics2Processor(BaseVisualRetrieverProcessor, Idefics2Processor):
    """
    Processor for ColIdefics2.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ColIdefics2 is deprecated and may be incompatible with future versions. Please use ColIdefics3 instead.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)

    def process_images(
        self,
        images: List[Image.Image],
        context_prompts: Optional[List[str]] = None,
    ) -> BatchEncoding:
        """
        Process images for ColIdefics2.

        Args:
            images: List of PIL images.
            context_prompts: List of optional context prompts, i.e. some text description of the context of the image.
        """

        texts_doc: List[str] = []
        images = [image.convert("RGB") for image in images]

        if not context_prompts:
            context_prompts = ["Describe the image."] * len(images)

        if len(images) != len(context_prompts):
            raise ValueError("Length of images and context prompts must match.")

        for context_prompt in context_prompts:
            messages_doc = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context_prompt},
                        {"type": "image"},
                    ],
                },
            ]
            text_doc = self.apply_chat_template(messages_doc, add_generation_prompt=False)
            texts_doc.append(text_doc.strip())

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
    ) -> BatchEncoding:
        """
        Process queries for ColIdefics2.
        """
        suffix = suffix or "<end_of_utterance>" * 5
        texts_query: List[str] = []

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
            text_query = self.apply_chat_template(messages_query, add_generation_prompt=False).strip()
            texts_query.append(text_query)

        batch_query = self(
            text=texts_query,
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

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
    ) -> Tuple[int, int]:
        raise NotImplementedError("This method is not implemented for ColIdefics2.")
