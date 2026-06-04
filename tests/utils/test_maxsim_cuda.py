"""CUDA-gated parity and training-smoke tests for the LIK MaxSim integration.

These tests verify that the fused kernel agrees with the pure-torch fallback on
both forward and backward (for both ``maxsim_inbatch`` and ``maxsim_kd``), and
that the migrated in-batch / negative-doc losses train without producing NaNs
or collapsing. They are marked ``@pytest.mark.slow`` so they are skipped by the
default CPU CI; run them with ``pytest -m slow`` on a host with a CUDA Ampere+
GPU and ``late-interaction-kernels`` installed.
"""

import pytest
import torch

from colpali_engine.loss.late_interaction_losses import (
    ColbertLoss,
    ColbertNegativeCELoss,
    ColbertPairwiseCELoss,
    ColbertPairwiseNegativeCELoss,
    ColbertSigmoidLoss,
)
from colpali_engine.utils._lik_backend import maxsim_inbatch_lik
from colpali_engine.utils.maxsim import (
    _torch_maxsim,
    _torch_maxsim_kd,
    maxsim_inbatch,
    maxsim_kd,
)

pytest.importorskip("late_interaction_kernels")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        reason="LIK kernel requires Ampere or newer",
    ),
]

EMBEDDING_DIM = 128
BATCH_SIZE = 4
QUERY_LEN = 16
DOC_LEN = 32
NUM_CANDIDATES = 4

_DTYPES: list[torch.dtype] = [torch.float32, torch.float16, torch.bfloat16]

# Tolerances calibrated to match the LIK project's own tests
# (see late-interaction-kernels/tests/test_padded.py).
_FORWARD_TOL: dict[torch.dtype, float] = {
    torch.float32: 5e-3,
    torch.float16: 2e-2,
    torch.bfloat16: 2e-2,
}
_BACKWARD_TOL: dict[torch.dtype, float] = {
    torch.float32: 1e-2,
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
}


def _random_inputs(dtype: torch.dtype, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    query = torch.randn(BATCH_SIZE, QUERY_LEN, EMBEDDING_DIM, dtype=dtype, device="cuda")
    doc = torch.randn(BATCH_SIZE, DOC_LEN, EMBEDDING_DIM, dtype=dtype, device="cuda")
    if requires_grad:
        query.requires_grad_(True)
        doc.requires_grad_(True)
    return query, doc


def _random_kd_inputs(dtype: torch.dtype, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(2)
    query = torch.randn(BATCH_SIZE, QUERY_LEN, EMBEDDING_DIM, dtype=dtype, device="cuda")
    doc = torch.randn(BATCH_SIZE, NUM_CANDIDATES, DOC_LEN, EMBEDDING_DIM, dtype=dtype, device="cuda")
    if requires_grad:
        query.requires_grad_(True)
        doc.requires_grad_(True)
    return query, doc


def test_strict_lik_mode_runs_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    # On an eligible GPU, strict mode must take the kernel, not raise LIKUnsupportedError.
    monkeypatch.setenv("COLPALI_SCORES_BACKEND", "lik")
    query, doc = _random_inputs(torch.float32)
    got = maxsim_inbatch(query, doc)
    assert torch.equal(got, maxsim_inbatch_lik(query, doc))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_forward_parity_cuda(dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    query, doc = _random_inputs(dtype)

    got = maxsim_inbatch(query, doc)

    monkeypatch.setenv("COLPALI_SCORES_BACKEND", "torch")
    want = maxsim_inbatch(query, doc)
    # Sanity: the disabled path equals the reference einsum exactly.
    assert torch.equal(want, _torch_maxsim(query, doc))

    assert got.shape == want.shape
    tol = _FORWARD_TOL[dtype]
    # LIK always returns fp32; cast the torch reference to fp32 too so the
    # comparison isn't bottlenecked by bf16/fp16 accumulation noise.
    torch.testing.assert_close(got.float(), want.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_backward_parity_cuda(dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    # LIK side: input dtype as-is, the kernel accumulates in fp32 internally.
    query_lik, doc_lik = _random_inputs(dtype, requires_grad=True)
    maxsim_inbatch(query_lik, doc_lik).sum().backward()

    # Reference side: cast the same input values to fp32 so the torch reduce
    # runs in fp32 too. Comparing against a bf16/fp16 torch reduce instead
    # would surface bf16/fp16 accumulation noise — not a kernel discrepancy.
    monkeypatch.setenv("COLPALI_SCORES_BACKEND", "torch")
    query_ref_dtype, doc_ref_dtype = _random_inputs(dtype)
    query_ref = query_ref_dtype.float().requires_grad_(True)
    doc_ref = doc_ref_dtype.float().requires_grad_(True)
    maxsim_inbatch(query_ref, doc_ref).sum().backward()

    tol = _BACKWARD_TOL[dtype]
    assert query_lik.grad is not None and query_ref.grad is not None
    assert doc_lik.grad is not None and doc_ref.grad is not None
    torch.testing.assert_close(query_lik.grad.float(), query_ref.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(doc_lik.grad.float(), doc_ref.grad, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_maxsim_kd_forward_parity_cuda(dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-query candidate lists route to LIK's kd_layout path; must match the einsum reference."""
    query, doc = _random_kd_inputs(dtype)

    got = maxsim_kd(query, doc)

    monkeypatch.setenv("COLPALI_SCORES_BACKEND", "torch")
    want = maxsim_kd(query, doc)
    # Sanity: the disabled path equals the reference einsum exactly.
    assert torch.equal(want, _torch_maxsim_kd(query, doc))

    assert got.shape == want.shape
    tol = _FORWARD_TOL[dtype]
    torch.testing.assert_close(got.float(), want.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_maxsim_kd_backward_parity_cuda(dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    """KD gradients from the LIK path must match an fp32 einsum reference."""
    query_lik, doc_lik = _random_kd_inputs(dtype, requires_grad=True)
    maxsim_kd(query_lik, doc_lik).sum().backward()

    # Reference path in fp32 — same input values cast up, einsum runs in fp32.
    monkeypatch.setenv("COLPALI_SCORES_BACKEND", "torch")
    query_ref_dtype, doc_ref_dtype = _random_kd_inputs(dtype)
    query_ref = query_ref_dtype.float().requires_grad_(True)
    doc_ref = doc_ref_dtype.float().requires_grad_(True)
    maxsim_kd(query_ref, doc_ref).sum().backward()

    tol = _BACKWARD_TOL[dtype]
    assert query_lik.grad is not None and query_ref.grad is not None
    assert doc_lik.grad is not None and doc_ref.grad is not None
    torch.testing.assert_close(query_lik.grad.float(), query_ref.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(doc_lik.grad.float(), doc_ref.grad, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "loss_cls",
    [ColbertLoss, ColbertPairwiseCELoss, ColbertSigmoidLoss],
)
def test_loss_training_smoke_cuda(loss_cls: type) -> None:
    """5 SGD steps on random embeddings — loss must stay finite and trend down."""
    torch.manual_seed(0)
    query = torch.randn(BATCH_SIZE, QUERY_LEN, EMBEDDING_DIM, device="cuda", requires_grad=True)
    doc = torch.randn(BATCH_SIZE, DOC_LEN, EMBEDDING_DIM, device="cuda", requires_grad=True)
    loss_fn = loss_cls(normalize_scores=False).to("cuda")
    optimizer = torch.optim.SGD([query, doc], lr=1e-2)

    losses: list[float] = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(query, doc)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses))), f"non-finite loss: {losses}"
    # Loose check: final loss should be at least as low as the initial. SGD on
    # a 4×16/4×32 random batch with lr=1e-2 reliably drives the loss down; the
    # weak inequality leaves headroom for stochastic edge cases.
    assert losses[-1] <= losses[0], f"loss did not decrease: {losses}"


@pytest.mark.parametrize(
    "loss_cls",
    [ColbertNegativeCELoss, ColbertPairwiseNegativeCELoss],
)
def test_negative_loss_training_smoke_cuda(loss_cls: type) -> None:
    """5 SGD steps through the negative-doc losses — exercises the maxsim_kd path."""
    torch.manual_seed(3)
    query = torch.randn(BATCH_SIZE, QUERY_LEN, EMBEDDING_DIM, device="cuda", requires_grad=True)
    doc = torch.randn(BATCH_SIZE, DOC_LEN, EMBEDDING_DIM, device="cuda", requires_grad=True)
    neg_doc = torch.randn(BATCH_SIZE, NUM_CANDIDATES, DOC_LEN, EMBEDDING_DIM, device="cuda", requires_grad=True)
    loss_fn = loss_cls(normalize_scores=False, in_batch_term_weight=0.0).to("cuda")
    optimizer = torch.optim.SGD([query, doc, neg_doc], lr=1e-2)

    losses: list[float] = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(query, doc, neg_doc)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses))), f"non-finite loss: {losses}"
    assert losses[-1] <= losses[0], f"loss did not decrease: {losses}"
