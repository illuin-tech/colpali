import torch
from torch.nn import CrossEntropyLoss


class BiEncoderModule(torch.nn.Module):
    """
    Base module for bi-encoder losses, handling buffer indexing and filtering hyperparameters.

    Args:
        max_batch_size (int): Maximum batch size for the pre-allocated index buffer.
        temperature (float): Scaling factor for logits (must be > 0).
        filter_threshold (float): Fraction of positive score above which negatives are down-weighted.
        filter_factor (float): Multiplicative factor applied to filtered negative scores.
    """

    def __init__(
        self,
        max_batch_size: int = 1024,
        temperature: float = 0.02,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be strictly positive")
        self.register_buffer("idx_buffer", torch.arange(max_batch_size), persistent=False)
        self.temperature = temperature
        self.filter_threshold = filter_threshold
        self.filter_factor = filter_factor

    def _get_idx(self, batch_size: int, offset: int, device: torch.device):
        """
        Generate index tensors for in-batch cross-entropy.

        Args:
            batch_size (int): Number of queries/docs in the batch.
            offset (int): Offset to apply for multi-GPU indexing.
            device (torch.device): Target device of the indices.

        Returns:
            Tuple[Tensor, Tensor]: (idx, pos_idx) both shape [batch_size].
        """
        idx = self.idx_buffer[:batch_size].to(device)
        return idx, idx + offset

    def _filter_high_negatives(self, scores: torch.Tensor, pos_idx: torch.Tensor):
        """
        In-place down-weighting of "too-high" in-batch negative scores.

        Args:
            scores (Tensor[B, B]): In-batch similarity matrix.
            pos_idx (Tensor[B]): Positive index for each query.
        """
        batch_size = scores.size(0)
        idx = self.idx_buffer[:batch_size].to(scores.device)
        pos_scores = scores[idx, pos_idx]
        thresh = self.filter_threshold * pos_scores.unsqueeze(1)
        mask = scores > thresh
        mask[idx, pos_idx] = False
        scores[mask] *= self.filter_factor


class BiEncoderLoss(BiEncoderModule):
    """
    InfoNCE loss for bi-encoders without explicit negatives.

    Args:
        temperature (float): Scaling factor for logits.
        pos_aware_negative_filtering (bool): Apply in-batch negative filtering if True.
        max_batch_size (int): Max batch size for index buffer caching.
        filter_threshold (float): Threshold ratio for negative filtering.
        filter_factor (float): Factor to down-weight filtered negatives.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, temperature, filter_threshold, filter_factor)
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.ce_loss = CrossEntropyLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Compute the InfoNCE loss over a batch of bi-encoder embeddings.

        Args:
            query_embeddings (Tensor[B, D]): Query vectors.
            doc_embeddings (Tensor[B, D]): Document vectors.
            offset (int): Offset for positive indices (multi-GPU).

        Returns:
            Tensor: Scalar cross-entropy loss.
        """
        # Compute in-batch similarity matrix
        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)
        batch_size = scores.size(0)
        idx, pos_idx = self._get_idx(batch_size, offset, scores.device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, pos_idx)

        return self.ce_loss(scores / self.temperature, pos_idx)


class BiNegativeCELoss(BiEncoderModule):
    """
    InfoNCE loss with explicit negative samples and optional in-batch term.

    Args:
        temperature (float): Scaling factor for logits.
        in_batch_term_weight (float): Weight for in-batch cross-entropy term (0 to 1).
        pos_aware_negative_filtering (bool): Apply in-batch negative filtering.
        max_batch_size (int): Max batch size for index buffer.
        filter_threshold (float): Threshold ratio for filtering.
        filter_factor (float): Factor to down-weight filtered negatives.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        in_batch_term_weight: float = 0.5,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, temperature, filter_threshold, filter_factor)
        self.in_batch_term_weight = in_batch_term_weight
        assert 0 <= in_batch_term_weight <= 1, "in_batch_term_weight must be between 0 and 1"
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.ce_loss = CrossEntropyLoss()
        # Inner InfoNCE for in-batch
        self.inner_loss = BiEncoderLoss(
            temperature=temperature,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Compute softplus(neg_score - pos_score) plus optional in-batch CE.

        Args:
            query_embeddings (Tensor[B, D]): Query vectors.
            doc_embeddings (Tensor[B, D]): Positive document vectors.
            neg_doc_embeddings (Tensor[B, D]): Negative document vectors.
            offset (int): Offset for in-batch CE positives.

        Returns:
            Tensor: Scalar loss value.
        """
        # Dot-product only for matching pairs
        pos_scores = (query_embeddings * doc_embeddings).sum(dim=1) / self.temperature
        neg_scores = (query_embeddings * neg_doc_embeddings).sum(dim=1) / self.temperature

        loss = torch.nn.functional.softplus(neg_scores - pos_scores).mean()

        if self.in_batch_term_weight > 0:
            loss_ib = self.inner_loss(query_embeddings, doc_embeddings, offset)
            loss = loss * (1 - self.in_batch_term_weight) + loss_ib * self.in_batch_term_weight
        return loss


class BiPairwiseCELoss(BiEncoderModule):
    """
    Pairwise softplus loss mining the hardest in-batch negative.

    Args:
        temperature (float): Scaling factor for logits.
        pos_aware_negative_filtering (bool): Filter high negatives before mining.
        max_batch_size (int): Maximum batch size for indexing.
        filter_threshold (float): Threshold for pos-aware filtering.
        filter_factor (float): Factor to down-weight filtered negatives.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, temperature, filter_threshold, filter_factor)
        self.pos_aware_negative_filtering = pos_aware_negative_filtering

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute softplus(hardest_neg - pos) where hardest_neg is the highest off-diagonal score.

        Args:
            query_embeddings (Tensor[B, D]): Query vectors.
            doc_embeddings (Tensor[B, D]): Document vectors.

        Returns:
            Tensor: Scalar loss value.
        """
        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)
        batch_size = scores.size(0)
        idx = self.idx_buffer[:batch_size].to(scores.device)
        pos = scores.diagonal()

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, idx)

        top2 = scores.topk(2, dim=1).values
        neg = torch.where(top2[:, 0] == pos, top2[:, 1], top2[:, 0])

        return torch.nn.functional.softplus((neg - pos) / self.temperature).mean()


class BiPairwiseNegativeCELoss(BiEncoderModule):
    """
    Pairwise softplus loss with explicit negatives and optional in-batch term.

    Args:
        temperature (float): Scaling factor for logits.
        in_batch_term_weight (float): Weight for in-batch cross-entropy term (0 to 1).
        max_batch_size (int): Maximum batch size for indexing.
        filter_threshold (float): Threshold for pos-aware filtering.
        filter_factor (float): Factor to down-weight filtered negatives.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        in_batch_term_weight: float = 0.5,
        max_batch_size: int = 1024,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, temperature, filter_threshold, filter_factor)
        self.in_batch_term_weight = in_batch_term_weight
        assert 0 <= in_batch_term_weight <= 1, "in_batch_term_weight must be between 0 and 1"
        self.inner_pairwise = BiPairwiseCELoss(
            temperature=temperature,
            pos_aware_negative_filtering=False,
            max_batch_size=max_batch_size,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute softplus(neg-explicit - pos) plus optional pairwise in-batch loss.

        Args:
            query_embeddings (Tensor[B, D]): Query vectors.
            doc_embeddings (Tensor[B, D]): Positive document vectors.
            neg_doc_embeddings (Tensor[B, D]): Negative document vectors.

        Returns:
            Tensor: Scalar loss value.
        """
        # dot product for matching pairs only
        pos = (query_embeddings * doc_embeddings).sum(dim=1)
        neg = (query_embeddings * neg_doc_embeddings).sum(dim=1)

        loss = torch.nn.functional.softplus((neg - pos) / self.temperature).mean()

        if self.in_batch_term_weight > 0:
            loss_ib = self.inner_pairwise(query_embeddings, doc_embeddings)
            loss = loss * (1 - self.in_batch_term_weight) + loss_ib * self.in_batch_term_weight

        return loss
