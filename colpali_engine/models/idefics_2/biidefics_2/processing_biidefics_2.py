from typing import List, Optional, Union

import torch

from colpali_engine.models.idefics_2.colidefics_2.processing_colidefics_2 import ColIdefics2Processor


class BiIdefics2Processor(ColIdefics2Processor):
    """
    Processor for BiIdefics2. Mirrors the `ColIdefics2Processor` class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        return self.score_single_vector(qs, ps, device=device)
