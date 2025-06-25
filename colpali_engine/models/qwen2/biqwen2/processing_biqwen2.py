from typing import List, Optional, Union

import torch
from transformers import BatchEncoding, BatchFeature

from colpali_engine.models.qwen2.colqwen2 import ColQwen2Processor


class BiQwen2Processor(ColQwen2Processor):
    """
    Processor for ColQwen2.
    """

    def process_texts(
        self,
        texts: List[str],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for ColQwen2.

        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
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
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_single_vector(qs, ps, device=device)
