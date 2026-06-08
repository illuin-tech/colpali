"""CPU-side tests for the late-interaction-kernels (LIK) dispatcher.

The CPU path always defers to the einsum reference, so the bulk of the suite verifies
the ``COLPALI_SCORES_BACKEND`` semantics (``auto`` falls back silently, ``torch`` forces
the reference, strict ``lik`` raises when the kernel can't run) and the end-to-end
parity between a dispatched call and a forced-torch call for both ``maxsim_inbatch``
and ``maxsim_kd``.
"""

from typing import Iterator

import pytest
import torch

from colpali_engine.utils._lik_backend import LIKUnsupportedError
from colpali_engine.utils.maxsim import (
    _resolve_backend,
    _torch_maxsim,
    _torch_maxsim_kd,
    maxsim_inbatch,
    maxsim_kd,
)

EMBEDDING_DIM = 32


@pytest.fixture
def clear_lik_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip the backend env var so the test sees the default ``auto`` mode."""
    monkeypatch.delenv("COLPALI_SCORES_BACKEND", raising=False)
    yield


class TestBackendResolution:
    def test_defaults_to_auto(self, clear_lik_env: None) -> None:
        assert _resolve_backend() == "auto"

    @pytest.mark.parametrize("backend", ["auto", "torch", "lik", "TORCH", "Lik"])
    def test_accepts_valid_backends_case_insensitively(self, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
        monkeypatch.setenv("COLPALI_SCORES_BACKEND", backend)
        assert _resolve_backend() == backend.lower()

    def test_rejects_unknown_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COLPALI_SCORES_BACKEND", "tensorrt")
        with pytest.raises(ValueError, match="COLPALI_SCORES_BACKEND"):
            _resolve_backend()


class TestStrictLikModeOnCpu:
    """``lik`` must raise on CPU inputs instead of silently falling back."""

    def test_maxsim_inbatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COLPALI_SCORES_BACKEND", "lik")
        query = torch.randn(2, 4, EMBEDDING_DIM)
        doc = torch.randn(3, 5, EMBEDDING_DIM)
        with pytest.raises(LIKUnsupportedError):
            maxsim_inbatch(query, doc)

    def test_maxsim_kd_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COLPALI_SCORES_BACKEND", "lik")
        query = torch.randn(2, 4, EMBEDDING_DIM)
        doc = torch.randn(2, 3, 5, EMBEDDING_DIM)
        with pytest.raises(LIKUnsupportedError):
            maxsim_kd(query, doc)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_maxsim_inbatch_matches_einsum_baseline(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    query = torch.randn(4, 6, EMBEDDING_DIM, dtype=dtype)
    doc = torch.randn(7, 10, EMBEDDING_DIM, dtype=dtype)

    got = maxsim_inbatch(query, doc)
    want = _torch_maxsim(query, doc)

    assert got.shape == (4, 7)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    assert torch.allclose(got, want, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_maxsim_kd_matches_einsum_baseline(dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    batch, num_candidates = 3, 5
    query = torch.randn(batch, 6, EMBEDDING_DIM, dtype=dtype)
    doc = torch.randn(batch, num_candidates, 8, EMBEDDING_DIM, dtype=dtype)

    got = maxsim_kd(query, doc)
    want = _torch_maxsim_kd(query, doc)

    assert got.shape == (batch, num_candidates)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    assert torch.allclose(got, want, atol=tol)


class TestEndToEndFallbackParity:
    """With LIK installed but CPU inputs, ``auto`` must take the reference path and
    produce identical results to a forced-torch run."""

    def test_maxsim_inbatch_cpu_matches_forced_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch.manual_seed(0)
        query = torch.randn(2, 4, EMBEDDING_DIM)
        doc = torch.randn(3, 5, EMBEDDING_DIM)

        monkeypatch.delenv("COLPALI_SCORES_BACKEND", raising=False)
        with_dispatcher = maxsim_inbatch(query, doc)

        monkeypatch.setenv("COLPALI_SCORES_BACKEND", "torch")
        forced_fallback = maxsim_inbatch(query, doc)

        torch.testing.assert_close(with_dispatcher, forced_fallback)

    def test_maxsim_kd_cpu_matches_forced_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch.manual_seed(1)
        batch, num_candidates = 3, 4
        query = torch.randn(batch, 5, EMBEDDING_DIM)
        doc = torch.randn(batch, num_candidates, 7, EMBEDDING_DIM)

        monkeypatch.delenv("COLPALI_SCORES_BACKEND", raising=False)
        with_dispatcher = maxsim_kd(query, doc)

        monkeypatch.setenv("COLPALI_SCORES_BACKEND", "torch")
        forced_fallback = maxsim_kd(query, doc)

        torch.testing.assert_close(with_dispatcher, forced_fallback)
