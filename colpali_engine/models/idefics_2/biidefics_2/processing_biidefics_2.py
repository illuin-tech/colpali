from typing import List, Optional
import torch

from colpali_engine.models.idefics_2.colidefics_2.processing_colidefics_2 import ColIdefics2Processor
from colpali_engine.utils.torch_utils import get_torch_device


class BiIdefics2Processor(ColIdefics2Processor):
    """
    Processor for BiIdefics2. Mirrors the `ColIdefics2Processor` class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_scores(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        device = device or get_torch_device()
        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        qs_stacked = torch.stack(qs).to(device)
        ps_stacked = torch.stack(ps).to(device)

        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
        return scores