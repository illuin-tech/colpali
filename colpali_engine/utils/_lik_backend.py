"""``late-interaction-kernels`` (LIK) backend for MaxSim scoring.

Lazily imported by ``colpali_engine.utils.maxsim`` so the dependency stays optional.
Mirrors PyLate's ``_lik_backend`` (lightonai/pylate#222): each entry point validates its
inputs and raises ``LIKUnsupportedError`` when the kernel cannot run, so the dispatcher can
fall back silently in ``auto`` mode and re-raise in strict ``lik`` mode.
"""

import torch

_IMPORT_OK: bool | None = None


class LIKUnsupportedError(Exception):
    """LIK cannot handle this call. The only exception ``auto`` mode swallows: real kernel errors propagate."""


def _lik_device(query: torch.Tensor, doc: torch.Tensor) -> str:
    """Validate the inputs against the kernel's constraints; return ``"cuda"`` or ``"mps"``."""
    if not is_available():
        raise LIKUnsupportedError(
            "late-interaction-kernels is not installed (pip install 'colpali-engine[lik]') "
            "or no CUDA/MPS accelerator is present."
        )
    if query.device != doc.device:
        raise LIKUnsupportedError(f"query and doc are on different devices ({query.device} vs {doc.device}).")
    # Below the Triton tile/MMA floor on the embedding dim the kernel can't beat einsum.
    if query.shape[-1] < 8:
        raise LIKUnsupportedError(f"embedding dim {query.shape[-1]} is below the kernel's tile floor (8).")
    if query.is_cuda:
        if torch.cuda.get_device_capability(query.device)[0] < 8:  # bf16 tensor cores need Ampere+
            raise LIKUnsupportedError("the CUDA kernel requires compute capability >= 8 (Ampere or newer).")
        return "cuda"
    if query.device.type == "mps":
        return "mps"
    raise LIKUnsupportedError(f"unsupported device type {query.device.type!r} (CUDA or MPS required).")


def is_available() -> bool:
    """Memoized: ``late_interaction_kernels`` imports and a CUDA/MPS accelerator is present."""
    global _IMPORT_OK
    if _IMPORT_OK is not None:
        return _IMPORT_OK
    try:
        import late_interaction_kernels  # noqa: F401
    except ImportError:
        _IMPORT_OK = False
        return _IMPORT_OK
    _IMPORT_OK = torch.cuda.is_available() or torch.backends.mps.is_available()
    return _IMPORT_OK


def maxsim_inbatch_lik(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """In-batch MaxSim through the fused kernel. Raises ``LIKUnsupportedError`` when ineligible."""
    device = _lik_device(query, doc)
    if device == "cuda":
        from late_interaction_kernels.autograd import maxsim

        return maxsim(query, doc)
    from late_interaction_kernels.mps import maxsim_mps

    return maxsim_mps(query, doc, normalize=False)


def maxsim_kd_lik(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """Per-query candidate MaxSim through the fused kernel. Raises ``LIKUnsupportedError`` when ineligible."""
    if _lik_device(query, doc) != "cuda":
        raise LIKUnsupportedError("the KD (candidate-list) layout has no MPS kernel.")
    if doc.dim() != 4:
        raise LIKUnsupportedError(f"KD layout expects doc [B, N, L, d], got {doc.dim()}-D.")

    from late_interaction_kernels.autograd import maxsim

    # A 4-D doc triggers the kernel's kd_layout path (one fused launch, no packing).
    return maxsim(query, doc)
