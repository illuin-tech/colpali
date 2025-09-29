from typing import List, Optional, Union

import torch
from transformers import BatchEncoding, BatchFeature

from colpali_engine.models.eurovbert.colvbert import ColEuroVBertProcessor


class BiVBertProcessor(ColEuroVBertProcessor):  # noqa: N801
    """
    Processor for BiVBert.
    """

    def process_texts(
        self,
        texts: List[str],
        max_length: int = 50,
        contexts: Optional[List[str]] = None,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for BiVBert.

        NOTE: `max_length` is not used and kept only for trainer compatibility.
        """
        if suffix is None:
            suffix = self.query_augmentation_token  # we remove buffer tokens
        if contexts is None:
            contexts = [""] * len(texts)

        prompts = [context + text + suffix for context, text in zip(contexts, texts)]

        batch_texts = self(
            text=prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=4096,
        )

        return batch_texts

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the cosine similarity for the given query and passage embeddings.
        """
        return self.score_single_vector(qs, ps, device=device)
