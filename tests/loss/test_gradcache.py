# ruff: noqa: N806, N812
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from colpali_engine.loss import (
    BiEncoderLoss,
    BiNegativeCELoss,
    ColbertLoss,
    ColbertNegativeCELoss,
    WithGradCache,
)
from colpali_engine.loss.gradcache import RandContext, _gather_doc_embeddings


class ToyEncoder(nn.Module):
    """Tiny deterministic encoder: returns (B, S, D) tokens, or (B, D) if pooled."""

    def __init__(self, vocab: int = 64, dim: int = 16, pooled: bool = False, dropout: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.lin = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.pooled = pooled

    def forward(self, input_ids, attention_mask=None):
        x = self.drop(self.lin(torch.tanh(self.emb(input_ids))))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        if self.pooled:
            denom = attention_mask.sum(1, keepdim=True).clamp(min=1).to(x.dtype)
            x = torch.nn.functional.normalize(x.sum(1) / denom, dim=-1)
        return x


def make_inputs(batch: int, num_neg: int = 0, seqlen: int = 12, vocab: int = 64, device="cpu"):
    """Build query/doc(/neg) inputs with full attention (lengths handled per-test)."""
    out = {}

    def field(prefix, extra=None):
        shape = (batch, seqlen) if extra is None else (batch, extra, seqlen)
        out[f"{prefix}_input_ids"] = torch.randint(1, vocab, shape, device=device)
        out[f"{prefix}_attention_mask"] = torch.ones(shape, dtype=torch.long, device=device)

    field("query")
    field("doc")
    if num_neg:
        field("neg_doc", extra=num_neg)
    return out


def reference_loss(model, loss_fn, inputs, num_neg, doc_embeddings=None, offset=0):
    """Standard non-cached path: full-batch forward, then loss.

    ``doc_embeddings``/``offset`` let a distributed caller substitute gathered docs.
    """
    q = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
    d = model(input_ids=inputs["doc_input_ids"], attention_mask=inputs["doc_attention_mask"])
    kwargs = dict(query_embeddings=q, doc_embeddings=doc_embeddings if doc_embeddings is not None else d, offset=offset)
    if num_neg:
        ids, am = inputs["neg_doc_input_ids"], inputs["neg_doc_attention_mask"]
        b, n, s = ids.shape
        neg = model(input_ids=ids.view(b * n, s), attention_mask=am.view(b * n, s))
        kwargs["neg_doc_embeddings"] = neg.view(b, n, *neg.shape[1:])
    return loss_fn(**kwargs), d, q.size(0)


def grads_of(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def max_rel_grad_diff(g1, g2):
    assert g1.keys() == g2.keys()
    worst = 0.0
    for n in g1:
        a, b = g1[n], g2[n]
        denom = a.abs().max().item() + 1e-12
        worst = max(worst, (a - b).abs().max().item() / denom)
    return worst


# Factories (not instances) so the loss can be rebuilt inside spawned workers.
def _bi_encoder():
    return BiEncoderLoss(temperature=1.0)


def _bi_negative():
    return BiNegativeCELoss(temperature=1.0, in_batch_term_weight=0.5)


def _colbert():
    return ColbertLoss(temperature=1.0, normalize_scores=False, use_smooth_max=False)


def _colbert_negative():
    return ColbertNegativeCELoss(
        temperature=1.0, normalize_scores=False, use_smooth_max=False, in_batch_term_weight=0.5
    )


_CASES = [  # (factory, pooled, num_neg, id)
    (_bi_encoder, True, 0, "BiEncoderLoss"),
    (_bi_negative, True, 3, "BiNegativeCELoss"),
    (_colbert, False, 0, "ColbertLoss"),
    (_colbert_negative, False, 3, "ColbertNegativeCELoss"),
]


class TestGradCacheEquivalence:
    """Single-process: GradCache must match the non-cached path bit-for-bit (up to fp noise)."""

    @pytest.mark.parametrize("mini_batch_size", [3, 8, 16], ids=["mbs<batch", "mbs=batch", "mbs>batch"])
    @pytest.mark.parametrize("factory,pooled,num_neg,name", _CASES, ids=[c[3] for c in _CASES])
    def test_loss_and_grad_match(self, factory, pooled, num_neg, name, mini_batch_size):
        batch = 8
        torch.manual_seed(0)
        model = ToyEncoder(pooled=pooled)
        torch.manual_seed(1)
        inputs = make_inputs(batch, num_neg=num_neg)
        loss_fn = factory()

        model.zero_grad(set_to_none=True)
        ref_loss, _, _ = reference_loss(model, loss_fn, inputs, num_neg)
        ref_loss.backward()
        ref_grads = grads_of(model)

        model.zero_grad(set_to_none=True)
        gc = WithGradCache(loss_fn, mini_batch_size=mini_batch_size)
        gc.gather_across_processes = False
        gc_loss = gc(model, inputs)
        gc_loss.backward()
        gc_grads = grads_of(model)

        assert torch.allclose(ref_loss, gc_loss, atol=1e-6, rtol=0), f"{name}: loss mismatch"
        assert max_rel_grad_diff(ref_grads, gc_grads) < 1e-4, f"{name}: grad mismatch"

    def test_eval_no_grad_path_returns_loss_only(self):
        """Under ``no_grad`` the wrapper returns the loss with no autograd hook."""
        batch = 6
        torch.manual_seed(0)
        model = ToyEncoder(pooled=False)
        torch.manual_seed(1)
        inputs = make_inputs(batch)
        loss_fn = _colbert()

        with torch.no_grad():
            ref_loss, _, _ = reference_loss(model, loss_fn, inputs, num_neg=0)
            gc = WithGradCache(loss_fn, mini_batch_size=4)
            gc.gather_across_processes = False
            gc_loss = gc(model, inputs)

        assert not gc_loss.requires_grad
        assert torch.allclose(ref_loss, gc_loss, atol=1e-6, rtol=0)

    def test_no_grad_does_not_touch_param_grads(self):
        torch.manual_seed(0)
        model = ToyEncoder(pooled=True)
        inputs = make_inputs(4)
        gc = WithGradCache(_bi_encoder(), mini_batch_size=2)
        gc.gather_across_processes = False
        with torch.no_grad():
            gc(model, inputs)
        assert all(p.grad is None for p in model.parameters())


class TestRandContext:
    """RandContext is what makes GradCache's two forward passes agree under dropout."""

    def test_dropout_is_reproduced(self):
        torch.manual_seed(0)
        model = ToyEncoder(pooled=False, dropout=0.5).train()
        ids = torch.randint(1, 64, (4, 10))
        mask = torch.ones(4, 10, dtype=torch.long)

        ctx = RandContext(ids, mask)
        with ctx:
            first = model(input_ids=ids, attention_mask=mask)
        # Without restoring RNG, a second dropout draw would differ; with the context it must not.
        with ctx:
            second = model(input_ids=ids, attention_mask=mask)
        assert torch.equal(first, second)

    def test_gradcache_with_dropout_is_deterministic_and_finite(self):
        """With dropout on, repeated GradCache calls under the same seed must agree exactly."""
        inputs = make_inputs(8, num_neg=0)

        def run():
            torch.manual_seed(0)
            model = ToyEncoder(pooled=False, dropout=0.3).train()
            torch.manual_seed(123)  # fix the stochastic forward
            gc = WithGradCache(_colbert(), mini_batch_size=3)
            gc.gather_across_processes = False
            model.zero_grad(set_to_none=True)
            loss = gc(model, inputs)
            loss.backward()
            return loss.detach(), grads_of(model)

        loss1, g1 = run()
        loss2, g2 = run()
        assert torch.isfinite(loss1) and all(torch.isfinite(g).all() for g in g1.values())
        assert torch.equal(loss1, loss2)
        assert max_rel_grad_diff(g1, g2) == 0.0


# Distributed (gloo / CPU) tests for the all-gather + offset path.
def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _dist_worker(rank, world_size, port, case, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        factory, pooled, num_neg, mbs, seqlen = case
        torch.manual_seed(0)  # identical params on every rank
        model = ToyEncoder(pooled=pooled)
        torch.manual_seed(100 + rank)  # rank-specific data (and per-rank seqlen)
        inputs = make_inputs(6, num_neg=num_neg, seqlen=seqlen[rank])
        loss_fn = factory()

        # Reference: gather docs across ranks exactly like the wrapper does, then full-batch loss.
        model.zero_grad(set_to_none=True)
        q = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
        d = model(input_ids=inputs["doc_input_ids"], attention_mask=inputs["doc_attention_mask"])
        d_gathered, offset = _gather_doc_embeddings(d, q.size(0))
        ref_loss, _, _ = reference_loss(model, loss_fn, inputs, num_neg, doc_embeddings=d_gathered, offset=offset)
        ref_loss.backward()
        ref_grads = grads_of(model)

        model.zero_grad(set_to_none=True)
        gc = WithGradCache(loss_fn, mini_batch_size=mbs)
        gc.gather_across_processes = True
        gc_loss = gc(model, inputs)
        gc_loss.backward()
        gc_grads = grads_of(model)

        results[rank] = {
            "loss_diff": (ref_loss - gc_loss).abs().item(),
            "grad_diff": max_rel_grad_diff(ref_grads, gc_grads),
            "error": None,
        }
    except Exception:
        import traceback

        results[rank] = {"loss_diff": None, "grad_diff": None, "error": traceback.format_exc()}
    finally:
        dist.destroy_process_group()


def _run_distributed(case, world_size=2):
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_dist_worker, args=(world_size, _find_free_port(), case, results), nprocs=world_size, join=True)
    return dict(results)


@pytest.mark.parametrize(
    "factory,pooled,num_neg,mbs,seqlen,name",
    [
        (_bi_encoder, True, 0, 4, (12, 12), "BiEncoder-equal-len"),
        (_bi_negative, True, 3, 4, (12, 12), "BiNegative-equal-len"),
        (_colbert, False, 0, 4, (12, 12), "Colbert-equal-len"),
        (_colbert, False, 0, 4, (10, 14), "Colbert-unequal-len-frontpad"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_gradcache_distributed_matches_reference(factory, pooled, num_neg, mbs, seqlen, name):
    """2-process gloo: GradCache == distributed non-cached path (gather + offset, incl. front-pad)."""
    results = _run_distributed((factory, pooled, num_neg, mbs, seqlen))
    assert len(results) == 2
    for rank, r in results.items():
        assert r["error"] is None, f"rank {rank} failed:\n{r['error']}"
        assert r["loss_diff"] < 1e-5, f"rank {rank} loss mismatch: {r['loss_diff']}"
        assert r["grad_diff"] < 1e-4, f"rank {rank} grad mismatch: {r['grad_diff']}"


# CUDA-only: autocast (mixed-precision) parity.
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_gradcache_autocast_parity_cuda(dtype):
    """The backward-hook re-forward runs outside autocast; the wrapper replays the policy itself."""
    batch = 8
    device = "cuda"
    torch.manual_seed(0)
    model = ToyEncoder(pooled=False).to(device)
    torch.manual_seed(1)
    inputs = make_inputs(batch, num_neg=0, device=device)
    loss_fn = _colbert()

    with torch.autocast(device_type="cuda", dtype=dtype):
        model.zero_grad(set_to_none=True)
        ref_loss, _, _ = reference_loss(model, loss_fn, inputs, num_neg=0)
    ref_loss.backward()
    ref_grads = grads_of(model)

    with torch.autocast(device_type="cuda", dtype=dtype):
        model.zero_grad(set_to_none=True)
        gc = WithGradCache(loss_fn, mini_batch_size=3)
        gc.gather_across_processes = False
        gc_loss = gc(model, inputs)
    gc_loss.backward()
    gc_grads = grads_of(model)

    # Mixed precision: compare loss within a loose tolerance and require finite, close grads.
    assert torch.allclose(ref_loss, gc_loss, atol=1e-2, rtol=1e-2)
    assert all(torch.isfinite(g).all() for g in gc_grads.values())
    assert max_rel_grad_diff(ref_grads, gc_grads) < 5e-2
