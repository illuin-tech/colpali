from typing import List, Optional, Union

import torch
from transformers import BatchEncoding, BatchFeature

from colpali_engine.models.qwen2_5.colqwen2_5 import ColQwen2_5_Processor


class BiQwen2_5_Processor(ColQwen2_5_Processor):  # noqa: N801
    """
    Processor for BiQwen2.5.
    """

    def process_texts(
        self,
        texts: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for BiQwen2.5.

        NOTE: `max_length` is not used and kept only for trainer compatibility.
        """
        if suffix is None:
            suffix = self.query_augmentation_token  # we remove buffer tokens

        prompts = [text + suffix for text in texts]

        batch_texts = self(
            text=prompts,
            return_tensors="pt",
            padding="longest",
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
