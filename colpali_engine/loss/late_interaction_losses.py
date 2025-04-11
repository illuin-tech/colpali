import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss


class ColbertLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.02, normalize_scores: bool = True):
        """
        InfoNCE loss generalized for late interaction models.
        Args:
            temperature: The temperature to use for the loss (`new_scores = scores / temperature`).
            normalize_scores: Whether to normalize the scores by the lengths of the query embeddings.
        """
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.temperature = temperature
        self.normalize_scores = normalize_scores

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)
        """

        scores = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)

        if self.normalize_scores:
            # find lengths of non-zero query embeddings
            # divide scores by the lengths of the query embeddings
            scores = scores / ((query_embeddings[:, :, 0] != 0).sum(dim=1).unsqueeze(-1))

            if not (scores >= 0).all().item() or not (scores <= 1).all().item():
                raise ValueError("Scores must be between 0 and 1 after normalization")

        loss_rowwise = self.ce_loss(scores / self.temperature, torch.arange(scores.shape[0], device=scores.device))

        return loss_rowwise


class ColbertNegativeCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.02, normalize_scores: bool = True, in_batch_term=False):
        """
        InfoNCE loss generalized for late interaction models with negatives.
        Args:
            temperature: The temperature to use for the loss (`new_scores = scores / temperature`).
            normalize_scores: Whether to normalize the scores by the lengths of the query embeddings.
            in_batch_term: Whether to include the in-batch term in the loss.
        """
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.in_batch_term = in_batch_term

    def forward(self, query_embeddings, doc_embeddings, neg_doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)
        neg_doc_embeddings: (batch_size, num_neg_doc_tokens, dim)
        """

        # Compute the ColBERT scores
        pos_scores = torch.einsum("bnd,bsd->bns", query_embeddings, doc_embeddings).max(dim=2)[0].sum(dim=1)
        neg_scores = torch.einsum("bnd,bsd->bns", query_embeddings, neg_doc_embeddings).max(dim=2)[0].sum(dim=1)

        loss = F.softplus(neg_scores / self.temperature - pos_scores / self.temperature).mean()

        if self.in_batch_term:
            scores = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
            if self.normalize_scores:
                # find lengths of non-zero query embeddings
                # divide scores by the lengths of the query embeddings
                scores = scores / ((query_embeddings[:, :, 0] != 0).sum(dim=1).unsqueeze(-1))

                if not (scores >= 0).all().item() or not (scores <= 1).all().item():
                    raise ValueError("Scores must be between 0 and 1 after normalization")
            loss += self.ce_loss(scores / self.temperature, torch.arange(scores.shape[0], device=scores.device))

        return loss / 2


class ColbertPairwiseCELoss(torch.nn.Module):
    def __init__(self):
        """
        Pairwise loss for ColBERT.
        """
        super().__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)

        Positive scores are the diagonal of the scores matrix.
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

        # Compute the loss
        # The loss is computed as the negative log of the softmax of the positive scores
        # relative to the negative scores.
        # This can be simplified to log-sum-exp of negative scores minus the positive score
        # for numerical stability.
        # torch.vstack((pos_scores, neg_scores)).T.softmax(1)[:, 0].log()*(-1)
        loss = F.softplus(neg_scores - pos_scores).mean()

        return loss


class ColbertPairwiseNegativeCELoss(torch.nn.Module):
    def __init__(self, in_batch_term=False):
        """
        Pairwise loss for ColBERT with negatives.
        Args:
            in_batch_term: Whether to include the in-batch term in the loss.
        """
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.in_batch_term = in_batch_term

    def forward(self, query_embeddings, doc_embeddings, neg_doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)
        neg_doc_embeddings: (batch_size, num_neg_doc_tokens, dim)
        """

        # Compute the ColBERT scores
        pos_scores = torch.einsum("bnd,bsd->bns", query_embeddings, doc_embeddings).max(dim=2)[0].sum(dim=1)
        neg_scores = torch.einsum("bnd,bsd->bns", query_embeddings, neg_doc_embeddings).max(dim=2)[0].sum(dim=1)

        loss = F.softplus(neg_scores - pos_scores).mean()

        if self.in_batch_term:
            scores = (
                torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
            )  # (batch_size, batch_size)

            # Positive scores are the diagonal of the scores matrix.
            pos_scores = scores.diagonal()  # (batch_size,)
            neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
            neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

            loss += F.softplus(neg_scores - pos_scores).mean()

        return loss / 2
