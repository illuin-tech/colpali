"""MaxSim dispatch: route late-interaction scoring through ``late-interaction-kernels`` (LIK)
or the pure-torch einsum reference, selected by ``COLPALI_SCORES_BACKEND``.

``maxsim_inbatch`` scores the in-batch ``[B, Lq, d] x [B, Ld, d]`` grid; ``maxsim_kd`` scores
the per-query candidate layout ``[B, N, Ld, d]`` used by the negative-doc losses.

``COLPALI_SCORES_BACKEND`` (read per call) mirrors PyLate's ``PYLATE_SCORES_BACKEND``:
``auto`` (default) uses LIK when eligible and silently falls back to torch, ``torch`` forces
the reference, ``lik`` requires the kernel and raises ``LIKUnsupportedError`` when it cannot run.

Pad tokens must be exactly zero: both paths rely on zero-padding, not an explicit mask.
"""

import os

import torch

_VALID_BACKENDS = ("auto", "torch", "lik")


def _resolve_backend() -> str:
    """Read ``COLPALI_SCORES_BACKEND`` at call time so it can be flipped at runtime."""
    backend = os.environ.get("COLPALI_SCORES_BACKEND", "auto").lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"COLPALI_SCORES_BACKEND must be one of {_VALID_BACKENDS}, got {backend!r}.")
    return backend


def _torch_maxsim(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """Reference in-batch MaxSim: ``einsum("bnd,csd->bcns").amax(-1).sum(-1)``."""
    return torch.einsum("bnd,csd->bcns", query, doc).amax(dim=3).sum(dim=2)


def _torch_maxsim_kd(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """Reference KD MaxSim over ``doc [B, N, L, d]`` (each query has its own N candidates)."""
    return torch.einsum("bnd,bksd->bkns", query, doc).amax(dim=3).sum(dim=2)


def maxsim_inbatch(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """In-batch MaxSim: ``[B_q, B_d]`` scores from zero-padded ``query [B_q, L_q, d]`` and ``doc [B_d, L_d, d]``."""
    backend = _resolve_backend()
    if backend != "torch":
        from colpali_engine.utils import _lik_backend

        try:
            return _lik_backend.maxsim_inbatch_lik(query, doc)
        except _lik_backend.LIKUnsupportedError:
            if backend == "lik":
                raise
    return _torch_maxsim(query, doc)


def maxsim_kd(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """Per-query candidate MaxSim: ``[B, N]`` from zero-padded ``query [B, L_q, d]`` and ``doc [B, N, L_d, d]``."""
    backend = _resolve_backend()
    if backend != "torch":
        from colpali_engine.utils import _lik_backend

        try:
            return _lik_backend.maxsim_kd_lik(query, doc)
        except _lik_backend.LIKUnsupportedError:
            if backend == "lik":
                raise
    return _torch_maxsim_kd(query, doc)
