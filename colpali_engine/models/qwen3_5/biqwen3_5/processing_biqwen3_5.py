from typing import List, Optional, Union

import torch
from transformers import BatchEncoding, BatchFeature

from colpali_engine.models.qwen3_5.colqwen3_5 import ColQwen3_5Processor


class BiQwen3_5Processor(ColQwen3_5Processor):  # noqa: N801
    """
    Processor for BiQwen3.5.
    """

    def process_texts(
        self,
        texts: List[str],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for BiQwen3.5.
        """
        return self(
            text=texts,
            return_tensors="pt",
            padding="longest",
        )

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
