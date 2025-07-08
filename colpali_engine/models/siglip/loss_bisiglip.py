import torch
from torch.nn import CrossEntropyLoss

from colpali_engine.loss.bi_encoder_losses import BiEncoderModule


class BiSigLipEncoderLoss(BiEncoderModule):
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
        model: any = None,
    ):
        super().__init__(max_batch_size, temperature, filter_threshold, filter_factor)
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.ce_loss = CrossEntropyLoss()
        self.model = model

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
        # scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)

        logits_per_text = torch.matmul(query_embeddings, doc_embeddings.t().to(doc_embeddings.device))

        logit_scale = self.model.logit_scale.to(query_embeddings.device)
        logit_bias = self.model.logit_bias.to(query_embeddings.device)
        scores = logits_per_text * logit_scale.exp() + logit_bias
        # scores = torch.sigmoid(scores)

        batch_size = scores.size(0)
        idx, pos_idx = self._get_idx(batch_size, offset, scores.device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, pos_idx)

        print(scores.shape, scores.argmax(dim=1), pos_idx)

        return self.ce_loss(scores / self.temperature, pos_idx)
