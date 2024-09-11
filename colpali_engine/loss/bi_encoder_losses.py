import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss


class BiEncoderLoss(torch.nn.Module):
    """
    Cross-entropy loss using the pairwise dot product scores between the query and document embeddings.
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
        - query_embeddings: (batch_size, dim)
        - doc_embeddings: (batch_size, dim)

        Returns:
        - torch.Tensor (1,)
        """
        if query_embeddings.shape[0] != doc_embeddings.shape[0]:
            raise ValueError("Batch size mismatch between query and document embeddings.")

        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)  # (batch_size, batch_size)
        loss_rowwise = self.ce_loss(scores, torch.arange(scores.shape[0], device=scores.device))  # (1,)

        return loss_rowwise


class BiPairwiseCELoss(torch.nn.Module):
    def __init__(self):
        """
        Compute the hard-margin loss using the pairwise dot product scores between the query and document embeddings.
        """
        super().__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
        - query_embeddings: (batch_size, dim)
        - doc_embeddings: (batch_size, dim)

        Returns:
        - torch.Tensor (1,)
        """
        if query_embeddings.shape[0] != doc_embeddings.shape[0]:
            raise ValueError("Batch size mismatch between query and document embeddings.")

        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)  # (batch_size, batch_size)

        # Positive scores are the diagonal of the scores matrix.
        pos_scores = scores.diagonal()  # (batch_size,)

        # Negative score for a given query is the maximum of the scores against all all other pages.
        # NOTE: We exclude the diagonal by setting it to a very low value. But since the embeddings are L2-normalized,
        # their maximum score is 1. Thus, we can subtract 1 from the diagonal to exclude it from the maximum operation.
        neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
        neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

        loss = F.softplus(neg_scores - pos_scores).mean()  # (1,)

        return loss


class BiPairwiseNegativeCELoss(torch.nn.Module):
    def __init__(self, in_batch_term=False):
        """
        Hard-margin loss using the pairwise dot product scores between:
        - the query and document embeddings
        - the query and negative document embeddings.
        """
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
        - query_embeddings: (batch_size, dim)
        - doc_embeddings: (batch_size, dim)
        - neg_doc_embeddings: (batch_size, dim)

        Returns:
        - torch.Tensor (1,)
        """

        # Compute the ColBERT scores
        pos_scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings).diagonal()  # (batch_size,)
        neg_scores = torch.einsum("bd,cd->bc", query_embeddings, neg_doc_embeddings).diagonal()  # (batch_size,)

        loss = F.softplus(neg_scores - pos_scores).mean()  # (1,)

        if self.in_batch_term:
            scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)  # (batch_size, batch_size)

            # Positive scores are the diagonal of the scores matrix.
            pos_scores = scores.diagonal()  # (batch_size,)

            neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
            neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

            loss += F.softplus(neg_scores - pos_scores).mean()  # (1,)

        return loss / 2
