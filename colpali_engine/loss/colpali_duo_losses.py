from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F  # noqa: N812

# from mpmath.libmp import normalize
from torch.nn import CrossEntropyLoss, KLDivLoss

# from triton.language.math import normcdf
from colpali_engine.loss.base_late_interaction_loss import BaseColbertLoss
from colpali_engine.models.paligemma.colpali_duo.modeling_colpali_duo import (
    ColPaliDuoLossOutputs,
    ColPaliDuoModelOutput,
)


@dataclass
class ColPaliDuoIntermediateLossOutputs:
    loss: torch.Tensor
    scores: Optional[torch.Tensor] = None


class ColPaliDuoLoss(BaseColbertLoss):
    """
    Loss function for ColPaliDuo.

    The loss function is a combination of two losses:
    1. Single-vector loss: Cross-entropy (with optional Matryoshka) loss between the query and document
        single-vector embeddings.
    2. Multi-vector loss: Margin loss between the query and document multi-vector embeddings.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        use_matryoshka_loss: bool = False,
        matryoshka_dims: Optional[List[int]] = None,
        matryoshka_weights: Optional[List[float]] = None,
        use_distillation_loss: bool = True,
        beta: float = 0.5,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.alpha = alpha

        # to test
        self.use_matryoshka_loss = use_matryoshka_loss
        self.matryoshka_dims = matryoshka_dims
        self.matryoshka_weights = matryoshka_weights

        self.use_distillation_loss = use_distillation_loss
        self.beta = beta
        self.temperature = temperature
        self.kl_div_loss = KLDivLoss(log_target=True)

    def single_vector_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        return_scores: bool = False,
    ) -> ColPaliDuoIntermediateLossOutputs:
        """
        Compute the loss function for the single-vector head.

        Args:
        - query_embeddings: (batch_size, dim)
        - doc_embeddings: (batch_size, dim)

        Returns:
        - ColPaliDuoIntermediateLossOutputs
        """

        if query_embeddings.shape[0] != doc_embeddings.shape[0]:
            raise ValueError("Batch size mismatch between query and document embeddings.")

        if query_embeddings.shape[1] != doc_embeddings.shape[1]:
            raise ValueError("Dimensionality mismatch between query and document embeddings.")

        batch_size = query_embeddings.shape[0]
        device = query_embeddings.device

        ce_loss_fn = CrossEntropyLoss()

        if not self.use_matryoshka_loss:
            scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)  # (batch_size, batch_size)
            # Positive scores are the diagonal of the scores matrix.
            pos_scores = scores.diagonal()  # (batch_size,)

            neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
            neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

            loss = F.softplus(neg_scores - pos_scores).mean()  # (1,)

            # # TODO - why modify the pairwise and be inconsistent ?
            # loss = ce_loss_fn.forward(
            #     input=scores,
            #     target=torch.arange(scores.shape[0], device=scores.device),
            # )  # ()

        else:
            if not self.matryoshka_dims:
                raise ValueError("Matryoshka dimensions must be provided when using Matryoshka loss.")

            # The target is independent of the Matryoshka dimensionality
            target = torch.arange(
                query_embeddings.shape[0],
                dtype=torch.long,
                device=query_embeddings.device,
            )

            # Initialize the scores matrix and the loss
            prev_scores = torch.zeros(batch_size, batch_size, device=device)
            scores = torch.zeros(batch_size, batch_size, device=device)
            loss = torch.tensor(0.0, device=device)  # ()

            # To efficiently compute the scores, we need the Matryoshka dimensions to be sorted.
            matryoshka_dims = [0] + sorted(self.matryoshka_dims)

            for prev_dim, dim in zip(matryoshka_dims, matryoshka_dims[1:]):
                scores = prev_scores + torch.einsum(
                    "bd,cd->bc", query_embeddings[:, prev_dim:dim], doc_embeddings[:, prev_dim:dim]
                )  # (batch_size, batch_size)
                loss += ce_loss_fn.forward(input=scores, target=target)  # ()
                prev_scores = scores

        return ColPaliDuoIntermediateLossOutputs(
            loss=loss,
            scores=scores if return_scores else None,
        )

    def multi_vector_loss(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        return_scores: bool = False,
    ) -> ColPaliDuoIntermediateLossOutputs:
        """
        Compute the loss function for the multi-vector head.

        Args:
        - query_embeddings: (batch_size, num_query_tokens, dim)
        - doc_embeddings: (batch_size, num_doc_tokens, dim)

        Returns:
        - ColPaliDuoIntermediateLossOutputs
        """

        if query_embeddings.shape[0] != doc_embeddings.shape[0]:
            raise ValueError("Batch size mismatch between query and document embeddings.")

        # Compute the ColBERT scores
        scores = self.compute_colbert_scores(query_embeddings, doc_embeddings)  # (batch_size, batch_size)

        # Positive scores are the diagonal of the scores matrix.
        pos_scores = scores.diagonal()  # (batch_size,)

        # Negative score for a given query is the maximum of the scores against all all other pages.
        # NOTE: We subtract a large value from the diagonal to exclude it from the maximum operation.
        neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
        neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

        # Compute the margin loss
        loss = F.softplus(neg_scores - pos_scores).mean().squeeze()  # ()

        return ColPaliDuoIntermediateLossOutputs(
            loss=loss,
            scores=scores if return_scores else None,
        )

    def distillation_loss(
        self,
        teacher_scores: torch.Tensor,
        student_scores: torch.Tensor,
    ) -> ColPaliDuoIntermediateLossOutputs:
        """
        Compute the distillation loss between the multi-vector head (teacher) and
        the single-vector head (student).

        Args:
        - teacher_scores: (batch_size, batch_size)
        - student_scores: (batch_size, batch_size)

        Returns:
        - ColPaliDuoIntermediateLossOutputs
        """

        if teacher_scores.shape != student_scores.shape:
            raise ValueError("Teacher and student scores should have the same shape.")

        # breakpoint()

        # Convert scores to probabilities
        teacher_probs = F.softmax(teacher_scores / self.temperature, dim=1)  # (batch_size, batch_size)
        student_log_probs = F.log_softmax(student_scores / self.temperature, dim=1)  # (batch_size, batch_size)

        # Compute KL divergence (input: log-probabilities, target: probabilities)
        loss_kd = self.temperature ** 2 * F.kl_div(
            input=student_log_probs,  # log-probabilities of student
            target=teacher_probs,  # probabilities of teacher
            reduction='batchmean'  # reduction is typically used to average the loss
        )


        return ColPaliDuoIntermediateLossOutputs(loss=loss_kd)

    def forward(
        self,
        query_embeddings: ColPaliDuoModelOutput,
        doc_embeddings: ColPaliDuoModelOutput,
    ) -> torch.Tensor: # ColPaliDuoLossOutputs:
        """
        Compute the total loss for the ColPaliDuo model.

        Args:
        - query_embeddings (ColPaliDuoModelOutput), all tensors with shape (batch_size, num_tokens, dim)
        - doc_embeddings (ColPaliDuoModelOutput), all tensors with shape (batch_size, num_tokens, dim)

        Returns:
        - ColPaliDuoLossOutputs
        """

        assert query_embeddings.single_vec_emb is not None
        assert doc_embeddings.single_vec_emb is not None

        single_vector_loss_outputs = self.single_vector_loss(
            query_embeddings.single_vec_emb,
            doc_embeddings.single_vec_emb,
            return_scores=True,
        )

        assert query_embeddings.multi_vec_emb is not None
        assert doc_embeddings.multi_vec_emb is not None

        multi_vector_loss_outputs = self.multi_vector_loss(
            query_embeddings.multi_vec_emb,
            doc_embeddings.multi_vec_emb,
            return_scores=True,
        )

        total_loss = self.alpha * single_vector_loss_outputs.loss + (1 - self.alpha) * multi_vector_loss_outputs.loss

        distillation_loss_outputs = None
        if self.use_distillation_loss:
            assert single_vector_loss_outputs.scores is not None
            assert multi_vector_loss_outputs.scores is not None

            # divide multi_vector_loss_outputs.scores by number of non-zero query_embeddings.multi_vec_emb for each row
            normalized_mv_scores = (multi_vector_loss_outputs.scores/
                                    (query_embeddings.multi_vec_emb.sum(dim=-1) != 0).sum(dim=-1).unsqueeze(1))
            distillation_loss_outputs = self.distillation_loss(
                normalized_mv_scores, single_vector_loss_outputs.scores,
            )
            total_loss += self.beta * distillation_loss_outputs.loss

        # would be nice to log the individual losses
        loss_dict =  ColPaliDuoLossOutputs(
            single_vector_loss=single_vector_loss_outputs.loss,
            multi_vector_loss=multi_vector_loss_outputs.loss,
            distillation_loss=distillation_loss_outputs.loss if distillation_loss_outputs is not None else None,
            loss=total_loss,
        )
        # log losses
        print(loss_dict)

        return total_loss
