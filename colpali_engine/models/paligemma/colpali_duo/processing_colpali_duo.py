from typing import List, Optional, Union

import torch

from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor


class ColPaliDuoProcessor(ColPaliProcessor):
    """
    Processor for ColPaliDuo. Mirrors the `ColPaliProcessor` class.

    TODO: duplicate transformers-ready code from https://github.com/huggingface/transformers/pull/33736 once merged
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = "single_vector"

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
        if self.mode == "single_vector":
            return self.score_single_vector(qs, ps, device=device)
        else:
            return self.score_multi_vector(qs, ps, device=device, **kwargs)
