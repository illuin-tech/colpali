# ruff: noqa: N806, N812
import pytest
import torch
import torch.nn.functional as F

from colpali_engine.loss import (
    ColbertLoss,
    ColbertModule,
    ColbertNegativeCELoss,
    ColbertPairwiseCELoss,
    ColbertPairwiseNegativeCELoss,
)


class TestColbertModule:
    def test_get_idx(self):
        module = ColbertModule(max_batch_size=5)
        idx, pos_idx = module._get_idx(batch_size=3, offset=2, device=torch.device("cpu"))
        assert torch.equal(idx, torch.tensor([0, 1, 2]))
        assert torch.equal(pos_idx, torch.tensor([2, 3, 4]))

    def test_smooth_max(self):
        module = ColbertModule(tau=2.0)
        scores = torch.tensor([[0.0, 2.0]])
        out = module._smooth_max(scores, dim=1)
        expected = 2.0 * torch.log(torch.tensor(1.0 + torch.exp(torch.tensor(1.0))))
        assert torch.allclose(out, expected)

    def test_apply_normalization_within_bounds(self):
        module = ColbertModule(norm_tol=1e-3)
        scores = torch.tensor([[0.5, 1.0], [0.2, 0.8]])
        lengths = torch.tensor([2.0, 4.0])
        normalized = module._apply_normalization(scores, lengths)
        expected = scores / lengths.unsqueeze(1)
        assert torch.allclose(normalized, expected)

    # def test_apply_normalization_out_of_bounds(self):
    #     module = ColbertModule(norm_tol=1e-3)
    #     scores = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    #     lengths = torch.tensor([1.0, 1.0])
    #     with pytest.raises(ValueError) as excinfo:
    #         module._apply_normalization(scores, lengths)
    #     assert "Scores out of bounds after normalization" in str(excinfo.value)

    def test_aggregate_max(self):
        module = ColbertModule()
        raw = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        out = module._aggregate(raw, use_smooth_max=False, dim_max=2, dim_sum=1)
        assert torch.allclose(out, torch.tensor([6.0, 14.0]))

    def test_aggregate_smooth_max(self):
        module = ColbertModule(tau=1.0)
        raw = torch.zeros(1, 2, 2)
        out = module._aggregate(raw, use_smooth_max=True, dim_max=2, dim_sum=1)
        assert torch.allclose(out, 2 * torch.log(torch.tensor(2.0)))

    def test_filter_high_negatives(self):
        module = ColbertModule(filter_threshold=0.95, filter_factor=0.5)
        scores = torch.tensor([[1.0, 0.96], [0.5, 1.0]])
        original = scores.clone()
        pos_idx = torch.tensor([0, 1])
        module._filter_high_negatives(scores, pos_idx)
        assert scores[0, 1] == pytest.approx(0.48)
        # other entries unchanged
        assert scores[0, 0] == original[0, 0]
        assert scores[1, 0] == original[1, 0]
        assert scores[1, 1] == original[1, 1]


class TestColbertLoss:
    def test_zero_embeddings(self):
        loss_fn = ColbertLoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
        )
        B, Nq, D = 3, 1, 4
        query = torch.zeros(B, Nq, D)
        doc = torch.zeros(B, Nq, D)
        loss = loss_fn(query, doc)
        expected = torch.log(torch.tensor(float(B)))
        assert torch.allclose(loss, expected)

    def test_with_and_without_filtering(self):
        base = ColbertLoss(
            temperature=1.0, normalize_scores=False, use_smooth_max=False, pos_aware_negative_filtering=False
        )
        filt = ColbertLoss(
            temperature=1.0, normalize_scores=False, use_smooth_max=False, pos_aware_negative_filtering=True
        )
        B, Nq, D = 2, 1, 3
        query = torch.zeros(B, Nq, D)
        doc = torch.zeros(B, Nq, D)
        assert torch.allclose(base(query, doc), filt(query, doc))


class TestColbertNegativeCELoss:
    def test_no_inbatch(self):
        loss_fn = ColbertNegativeCELoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
            in_batch_term_weight=0,
        )
        B, Lq, D, Lneg, Nneg = 2, 1, 3, 1, 1
        query = torch.zeros(B, Lq, D)
        doc = torch.zeros(B, Lq, D)
        neg = torch.zeros(B, Nneg, Lneg, D)
        loss = loss_fn(query, doc, neg)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)

    def test_with_inbatch(self):
        loss_fn = ColbertNegativeCELoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
            in_batch_term_weight=0.5,
        )
        B, Lq, D, Lneg, Nneg = 2, 1, 3, 1, 1
        query = torch.zeros(B, Lq, D)
        doc = torch.zeros(B, Lq, D)
        neg = torch.zeros(B, Nneg, Lneg, D)
        loss = loss_fn(query, doc, neg)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)


class TestColbertPairwiseCELoss:
    def test_zero_embeddings(self):
        loss_fn = ColbertPairwiseCELoss(
            temperature=1.0, normalize_scores=False, use_smooth_max=False, pos_aware_negative_filtering=False
        )
        B, Nq, D = 2, 1, 3
        query = torch.zeros(B, Nq, D)
        doc = torch.zeros(B, Nq, D)
        loss = loss_fn(query, doc)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)


class TestColbertPairwiseNegativeCELoss:
    def test_no_inbatch(self):
        loss_fn = ColbertPairwiseNegativeCELoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
            in_batch_term_weight=0,
        )
        B, Lq, D, Lneg, Nneg = 2, 1, 3, 1, 1
        query = torch.zeros(B, Lq, D)
        doc = torch.zeros(B, Lq, D)
        neg = torch.zeros(B, Nneg, Lneg, D)
        loss = loss_fn(query, doc, neg)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)

    def test_with_inbatch(self):
        loss_fn = ColbertPairwiseNegativeCELoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
            in_batch_term_weight=0.5,
        )
        B, Lq, D, Lneg, Nneg = 2, 1, 3, 1, 1
        query = torch.zeros(B, Lq, D)
        doc = torch.zeros(B, Lq, D)
        neg = torch.zeros(B, Nneg, Lneg, D)
        loss = loss_fn(query, doc, neg)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)
