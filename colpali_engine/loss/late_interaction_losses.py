import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss

from colpali_engine.loss.base_late_interaction_loss import BaseColbertLoss


class ColbertCELoss(BaseColbertLoss):
    """
    Cross-entropy loss using the ColBERT scores between the query and document embeddings.
    """

    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
        - query_embeddings: (batch_size, num_query_tokens, dim)
        - doc_embeddings: (batch_size, num_doc_tokens, dim)

        Returns:
        - torch.Tensor (1,)
        """

        if query_embeddings.shape[0] != doc_embeddings.shape[0]:
            raise ValueError("Batch size mismatch between query and document embeddings.")

        scores = self.compute_colbert_scores(qs=query_embeddings, ps=doc_embeddings)  # (batch_size, batch_size)

        loss_rowwise = self.ce_loss(scores, torch.arange(scores.shape[0], device=scores.device))  # (1,)

        return loss_rowwise


class ColbertPairwiseCELoss(BaseColbertLoss):
    """
    Hard-margin loss using the ColBERT scores between the query and document embeddings.
    """

    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
        - query_embeddings: (batch_size, num_query_tokens, dim)
        - doc_embeddings: (batch_size, num_doc_tokens, dim)

        Returns:
        - torch.Tensor (1,)
        """

        if query_embeddings.shape[0] != doc_embeddings.shape[0]:
            raise ValueError("Batch size mismatch between query and document embeddings.")

        # Compute the ColBERT scores
        scores = self.compute_colbert_scores(qs=query_embeddings, ps=doc_embeddings)  # (batch_size, batch_size)

        # Positive scores are the diagonal of the scores matrix
        pos_scores = scores.diagonal()  # (batch_size,)

        # Negative score for a given query is the maximum of the scores against all all other pages.
        # NOTE: We subtract a large value from the diagonal to exclude it from the maximum operation.
        neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
        neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

        # Compute the loss
        # The loss is computed as the negative log of the softmax of the positive scores relative to the negative
        # scores. This can be simplified to log-sum-exp of negative scores minus the positive score for numerical
        # stability.
        loss = F.softplus(neg_scores - pos_scores).mean()

        return loss


class ColbertPairwiseNegativeCELoss(BaseColbertLoss):
    """
    Hard-margin loss using the ColBERT scores between:
        - the query and the document embeddings
        - the query and the negative document embeddings.
    """

    def __init__(self, in_batch_term: bool = False):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.in_batch_term = in_batch_term

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
        - query_embeddings: (batch_size, num_query_tokens, dim)
        - doc_embeddings: (batch_size, num_doc_tokens, dim)
        - neg_doc_embeddings: (batch_size, num_doc_tokens, dim)

        Returns:
        - torch.Tensor (1,)
        """

        if query_embeddings.shape[0] != doc_embeddings.shape[0]:
            raise ValueError("Batch size mismatch between query and document embeddings.")

        # Compute the ColBERT scores
        pos_scores = self.compute_colbert_scores(qs=query_embeddings, ps=doc_embeddings)  # (batch_size, batch_size)
        neg_scores = self.compute_colbert_scores(qs=query_embeddings, ps=neg_doc_embeddings)  # (batch_size, batch_size)

        loss = F.softplus(neg_scores - pos_scores).mean()  # (1,)

        if self.in_batch_term:
            scores = self.compute_colbert_scores(qs=query_embeddings, ps=doc_embeddings)  # (batch_size, batch_size)

            # Positive scores are the diagonal of the scores matrix.
            pos_scores = scores.diagonal()  # (batch_size,)
            neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
            neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

            loss += F.softplus(neg_scores - pos_scores).mean()  # (1,)

        return loss / 2
