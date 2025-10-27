# ruff: noqa: N806, N812
import pytest
import torch
import torch.nn.functional as F

from colpali_engine.loss import (
    BiEncoderLoss,
    BiEncoderModule,
    BiNegativeCELoss,
    BiPairwiseCELoss,
    BiPairwiseNegativeCELoss,
)


class TestBiEncoderModule:
    def test_init_invalid_temperature(self):
        with pytest.raises(ValueError) as excinfo:
            BiEncoderModule(temperature=0.0)
        assert "Temperature must be strictly positive" in str(excinfo.value)

    def test_get_idx(self):
        module = BiEncoderModule(max_batch_size=5, temperature=0.1)
        idx, pos_idx = module._get_idx(batch_size=3, offset=2, device=torch.device("cpu"))
        assert torch.equal(idx, torch.tensor([0, 1, 2]))
        assert torch.equal(pos_idx, torch.tensor([2, 3, 4]))

    def test_filter_high_negatives(self):
        module = BiEncoderModule(filter_threshold=0.95, filter_factor=0.5, temperature=0.1)
        # Create a 2×2 score matrix where scores[0,1] > 0.95 * pos_score[0]
        scores = torch.tensor([[1.0, 0.96], [0.5, 1.0]])
        original = scores.clone()
        pos_idx = torch.tensor([0, 1])
        module._filter_high_negatives(scores, pos_idx)
        # Only scores[0,1] should be down-weighted
        assert scores[0, 1] == pytest.approx(0.48)
        # Other entries unchanged
        assert scores[0, 0] == original[0, 0]
        assert scores[1, 0] == original[1, 0]
        assert scores[1, 1] == original[1, 1]


class TestBiEncoderLoss:
    def test_forward_zero_embeddings(self):
        loss_fn = BiEncoderLoss(temperature=1.0, pos_aware_negative_filtering=False)
        B, D = 4, 5
        query = torch.zeros(B, D)
        doc = torch.zeros(B, D)
        loss = loss_fn(query, doc)
        # scores are all zeros => uniform softmax => loss = log(B)
        expected = torch.log(torch.tensor(float(B)))
        assert torch.allclose(loss, expected)

    def test_forward_with_filtering(self):
        loss_fn = BiEncoderLoss(temperature=1.0, pos_aware_negative_filtering=True)
        B, D = 3, 2
        query = torch.zeros(B, D)
        doc = torch.zeros(B, D)
        # Filtering on zero scores should have no effect
        loss1 = loss_fn(query, doc)
        loss2 = BiEncoderLoss(temperature=1.0, pos_aware_negative_filtering=False)(query, doc)
        assert torch.allclose(loss1, loss2)


class TestBiNegativeCELoss:
    def test_forward_no_inbatch(self):
        loss_fn = BiNegativeCELoss(temperature=1.0, in_batch_term_weight=0, pos_aware_negative_filtering=False)
        B, D, Nneg = 3, 4, 1
        query = torch.zeros(B, D)
        pos = torch.zeros(B, D)
        neg = torch.zeros(B, Nneg, D)
        loss = loss_fn(query, pos, neg)
        # softplus(0 - 0) = ln(2)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)

    def test_forward_with_inbatch(self):
        loss_fn = BiNegativeCELoss(temperature=1.0, in_batch_term_weight=0.5, pos_aware_negative_filtering=False)
        B, D, Nneg = 2, 3, 1
        query = torch.zeros(B, D)
        pos = torch.zeros(B, D)
        neg = torch.zeros(B, Nneg, D)
        loss = loss_fn(query, pos, neg)
        # in-batch CE on zeros: log(B)
        ce = torch.log(torch.tensor(float(B)))
        sp = F.softplus(torch.tensor(0.0))
        expected = (sp + ce) / 2
        assert torch.allclose(loss, expected)


class TestBiPairwiseCELoss:
    def test_forward_zero_embeddings(self):
        loss_fn = BiPairwiseCELoss(temperature=1.0, pos_aware_negative_filtering=False)
        B, D = 4, 6
        query = torch.zeros(B, D)
        doc = torch.zeros(B, D)
        loss = loss_fn(query, doc)
        # hardest neg = 0, pos = 0 => softplus(0) = ln(2)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)

    def test_forward_with_filtering(self):
        loss_fn = BiPairwiseCELoss(temperature=1.0, pos_aware_negative_filtering=True)
        B, D = 3, 5
        query = torch.zeros(B, D)
        doc = torch.zeros(B, D)
        # Filtering on zero scores should not change result
        assert torch.allclose(loss_fn(query, doc), BiPairwiseCELoss(temperature=1.0)(query, doc))


class TestBiPairwiseNegativeCELoss:
    def test_forward_no_inbatch(self):
        loss_fn = BiPairwiseNegativeCELoss(temperature=1.0, in_batch_term_weight=0)
        B, Nneg, D = 5, 2, 4
        query = torch.zeros(B, D)
        pos = torch.zeros(B, D)
        neg = torch.zeros(B, Nneg, D)
        loss = loss_fn(query, pos, neg)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)

    def test_forward_with_inbatch(self):
        loss_fn = BiPairwiseNegativeCELoss(temperature=1.0, in_batch_term_weight=0.5)
        B, Nneg, D = 2, 3, 4
        query = torch.zeros(B, D)
        pos = torch.zeros(B, D)
        neg = torch.zeros(B, Nneg, D)
        loss = loss_fn(query, pos, neg)
        # both explicit and in-batch pairwise yield ln(2), average remains ln(2)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)
