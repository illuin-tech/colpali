import logging
from typing import List

import torch

from colpali_engine.utils.torch_utils import get_torch_device

logger = logging.getLogger(__name__)


class RetrievalScorer:
    """
    Scorer for retrieval tasks. Supports both single-vector and multi-vector embeddings.
    """

    def __init__(
        self,
        is_multi_vector: bool = False,
        device: str = "auto",
    ):
        self.is_multi_vector = is_multi_vector
        self.device = get_torch_device(device)

    def evaluate(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the retrieval scores for the given query and passage embeddings. Uses the scoring method
        based on the `is_multi_vector` attribute.
        """
        if self.is_multi_vector:
            scores = self._get_multi_vector_scores(qs, ps)
        else:
            scores = self._get_single_vector_scores(qs, ps)

        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def compute_top_1_accuracy(scores: torch.Tensor) -> float:
        """
        Compute the top-1 accuracy from the given scores.
        """
        arg_score = scores.argmax(dim=1)
        accuracy = (arg_score == torch.arange(scores.shape[0], device=scores.device)).sum().item() / scores.shape[0]
        return accuracy

    def _get_single_vector_scores(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        if len(qs) == 0:
            raise ValueError("No querie(s) provided")
        if len(ps) == 0:
            raise ValueError("No passage(s) provided")

        qs_stacked = torch.stack(qs).to(self.device)
        ps_stacked = torch.stack(ps).to(self.device)

        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
        return scores

    def _get_multi_vector_scores(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        batch_size: int = 128,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        if len(qs) == 0:
            raise ValueError("No querie(s) provided")
        if len(ps) == 0:
            raise ValueError("No passage(s) provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                self.device
            )
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to(self.device)
                scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        return scores
