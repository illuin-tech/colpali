import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss


class ColbertLoss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max=False,
        pos_aware_negative_filtering: bool = False,
    ):
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
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering

    def forward(self, query_embeddings, doc_embeddings, offset: int = 0):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)
        offset: The offset to use for the loss. This is useful in cross-gpu tasks when there are more docs than queries.
        """

        scores = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)

        if self.use_smooth_max:
            # τ is a temperature hyperparameter (smaller τ → closer to hard max)
            tau = 0.1
            # logsumexp gives a smooth approximation of max
            soft_max = tau * torch.logsumexp(scores / tau, dim=3)  # shape [b, c, n]
            scores = soft_max.sum(dim=2)
        else:
            scores = scores.amax(dim=3).sum(dim=2)

        if self.normalize_scores:
            # find lengths of non-zero query embeddings
            # divide scores by the lengths of the query embeddings
            scores = scores / ((query_embeddings[:, :, 0] != 0).sum(dim=1).unsqueeze(-1))

            if not self.use_smooth_max and (not (scores >= 0).all().item() or not (scores <= 1).all().item()):
                raise ValueError("Scores must be between 0 and 1 after normalization")

        batch_size = scores.size(0)
        idx = torch.arange(batch_size, device=scores.device)
        pos_idx = idx + offset

        acc = (scores.argmax(dim=1) == pos_idx).sum().item() / batch_size

        if self.pos_aware_negative_filtering:
            pos_scores = scores[idx, pos_idx]  # shape [B]
            # 1) build a mask of “too-high” negatives
            thresholds = 0.95 * pos_scores.unsqueeze(1)  # shape [B,1]
            high_neg_mask = scores > thresholds  # shape [B,B]
            high_neg_mask[idx, pos_idx] = False
            scores[high_neg_mask] *= 0.5
            print(f"Acc: {acc}, num_high_neg per row: {high_neg_mask.sum().item() / high_neg_mask.shape[0]}")

        loss_rowwise = self.ce_loss(scores / self.temperature, pos_idx)
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

    def forward(self, query_embeddings, doc_embeddings, neg_doc_embeddings, offset: int = 0):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)
        neg_doc_embeddings: (batch_size, num_neg_doc_tokens, dim)
        offset: The offset to use for the loss. This is useful in cross-gpu tasks when there are more docs than queries.
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
            loss += self.ce_loss(
                scores / self.temperature, torch.arange(scores.shape[0], device=scores.device) + offset
            )

        return loss / 2


class ColbertPairwiseCELoss(torch.nn.Module):
    def __init__(self, use_smooth_max=False):
        """
        Pairwise loss for ColBERT.
        """
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.use_smooth_max = use_smooth_max

    def forward(self, query_embeddings, doc_embeddings, offset: int = 0):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)

        Positive scores are the diagonal of the scores matrix.
        """

        # Compute the ColBERT scores
        scores = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)  # (batch_size, batch_size)

        if self.use_smooth_max:
            # τ is a temperature hyperparameter (smaller τ → closer to hard max)
            tau = 0.1
            # logsumexp gives a smooth approximation of max
            soft_max = tau * torch.logsumexp(scores / tau, dim=3)
            scores = soft_max.sum(dim=2)
        else:
            scores = scores.max(dim=3)[0].sum(dim=2)

        # Positive scores are the diagonal of the scores matrix but shifted by the offset.
        pos_scores = scores.diagonal(offset=offset)  # (batch_size,)

        # 1) clone so you don’t overwrite your original scores
        neg_scores = scores.clone()
        # 2) get a *view* on that offset‐diagonal…
        d = neg_scores.diagonal(offset=offset)
        # 3) fill it with a very large negative (or -inf) so it never wins the max
        d.fill_(-1e6)

        if self.use_smooth_max:
            # τ is a temperature hyperparameter (smaller τ → closer to hard max)
            tau = 0.1
            # logsumexp gives a smooth approximation of max
            neg_scores = tau * torch.logsumexp(neg_scores / tau, dim=1)
        else:
            neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

        # Compute the loss
        # The loss is computed as the negative log of the softmax of the positive scores
        # relative to the negative scores.
        # This can be simplified to log-sum-exp of negative scores minus the positive score
        # for numerical stability.
        # torch.vstack((pos_scores, neg_scores)).T.softmax(1)[:, 0].log()*(-1)
        loss = F.softplus(neg_scores - pos_scores).mean()

        import torch.distributed as dist

        print(
            f"Scores (0): {pos_scores[0].item(), neg_scores[0].item()}, shapes: {scores.shape}, {pos_scores.shape}, {neg_scores.shape}"
        )
        print(
            f"Rank: {dist.get_rank()}, Offset: {offset}, acc: {(scores.argmax(dim=1) == torch.arange(scores.shape[0], device=scores.device) + offset).sum().item() / scores.shape[0]},  scores: {scores.argmax(dim=1)}"
        )

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
