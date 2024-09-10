from typing import List, Optional
import torch
from PIL import Image
from transformers import BatchEncoding, Idefics2Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import get_torch_device

class ColIdefics2Processor(BaseVisualRetrieverProcessor, Idefics2Processor):
    """
    Processor for ColIdefics2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchEncoding:
        """
        Process images for ColIdefics2.
        """
        texts_doc: List[str] = []
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


    @staticmethod
    def _get_scores(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        batch_size: int = 128,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        device = device or get_torch_device(device)
        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs[i : i + batch_size], batch_first=True, padding_value=0
            ).to(device)
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to(device)
                scores_batch.append(
                    torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                    .max(dim=3)[0]
                    .sum(dim=2)
                )
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        return scores