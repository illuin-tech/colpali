from typing import List, Optional, Union

import torch
from transformers import BatchEncoding, BatchFeature

from .processing_colqwen3_omni import ColQwen3OmniProcessor


class BiQwen3OmniProcessor(ColQwen3OmniProcessor):
    """
    Processor for BiQwen3-Omni / BidirLM-Omni checkpoints.
    """

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
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
        return self.score_single_vector(qs, ps, device=device)
