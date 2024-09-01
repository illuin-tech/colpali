from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F  # noqa: N812

from colpali_engine.models.colpali_2.colpali_2_modeling_outputs import ColPali2ModelOutput


class MatryoshkaCELoss(torch.nn.Module):
    """
    Loss function for Matryoshka Representation Learning
    Adapted from https://github.com/RAIVNLab/MRL/blob/7ccb42df6be05f3d21d0648aa03099bba46386bf/MRL.py#L11
    """

    def __init__(self, relative_importance: Optional[List[float]] = None, **kwargs):
        super(MatryoshkaCELoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(**kwargs)
        self.relative_importance = relative_importance

    def forward(self, output, target) -> torch.Tensor:
        # Calculate losses for each output and stack them. This is still O(N)
        losses = torch.stack([self.criterion(output_i, target) for output_i in output])

        # Set relative_importance to 1 if not specified
        rel_importance = (
            torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance)
        )

        # Apply relative importance weights
        weighted_losses = rel_importance * losses
        return weighted_losses.sum()


@dataclass
class ColPali2LossOutputs:
    single_vector_loss: torch.Tensor
    multi_vector_loss: torch.Tensor
    total_loss: torch.Tensor


class ColPali2Loss(torch.nn.Module):
    """
    Loss function for ColPali2.
    The loss function is a combination of two losses:
    1. Single-vector loss: Cross-entropy (with optional Matryoshka) loss between the query and document
        single-vector embeddings.
    2. Multi-vector loss: Margin loss between the query and document multi-vector embeddings.
    """

    def __init__(
        self,
        use_matryoshka_loss: bool = True,
    ):
        super().__init__()
        self.use_matryoshka_loss = use_matryoshka_loss
        self.alpha: float = 0.5

    def single_vector_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        query_embeddings: (batch_size, dim)
        doc_embeddings: (batch_size, dim)
        """
        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)

        if self.use_matryoshka_loss:
            loss = self.single_vector_loss(scores, torch.arange(scores.shape[0], device=scores.device))
        else:
            loss = F.cross_entropy(scores, torch.arange(scores.shape[0], device=scores.device))
        return loss

    def multi_vector_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)
        """
        # Compute the ColBERT scores
        scores = (
            torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
        )  # (batch_size, batch_size)

        # Positive scores are the diagonal of the scores matrix.
        pos_scores = scores.diagonal()  # (batch_size,)

        # Negative score for a given query is the maximum of the scores against all all other pages.
        # NOTE: We exclude the diagonal by setting it to a very low value: since we know the maximum score is 1,
        # we can subtract 1 from the diagonal to exclude it from the maximum operation.
        neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
        neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

        # Compute the margin loss
        loss = F.softplus(neg_scores - pos_scores).mean()

        return loss

    def forward(
        self,
        query_embeddings: ColPali2ModelOutput,
        doc_embeddings: ColPali2ModelOutput,
    ) -> ColPali2LossOutputs:
        single_vector_loss = self.single_vector_loss(query_embeddings.single_vec_emb, doc_embeddings.single_vec_emb)
        multi_vector_loss = self.multi_vector_loss(query_embeddings.multi_vec_emb, doc_embeddings.multi_vec_emb)

        total_loss = self.alpha * single_vector_loss + (1 - self.alpha) * multi_vector_loss

        return ColPali2LossOutputs(single_vector_loss, multi_vector_loss, total_loss)
