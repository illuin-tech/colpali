import contextlib
from functools import partial
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import tqdm
from torch.distributed.nn.functional import all_gather
from torch.utils.checkpoint import get_device_states, set_device_states


class RandContext:
    """
    Captures the RNG state so a forward pass can be replayed deterministically.

    GradCache runs every forward pass twice (once without gradients to build the cache,
    once with gradients in the backward hook). Restoring the RNG state guarantees that
    stochastic ops such as dropout produce identical outputs across both passes.
    """

    def __init__(self, *tensors: torch.Tensor) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _gather_doc_embeddings(doc_embeddings: torch.Tensor, local_batch_size: int) -> Tuple[torch.Tensor, int]:
    """
    Only positive documents are gathered, so each rank can use other ranks' positives as
    additional in-batch negatives. Explicit hard negatives are intentionally not gathered
    (they are consumed locally), matching the standard non-cached loss path.

    Returns the gathered embeddings and the offset of this rank's positives inside them.
    Token-level (late-interaction) embeddings carry a sequence dim that may differ across
    ranks, so it is padded to the global max; pooled (bi-encoder) embeddings need no padding.
    """
    if not _is_distributed() or local_batch_size <= 0:
        return doc_embeddings, 0

    # Late-interaction embeddings are (batch, seq_len, dim); pad the variable sequence length.
    if doc_embeddings.dim() == 3:
        max_len = torch.tensor(doc_embeddings.size(1), device=doc_embeddings.device)
        torch.distributed.all_reduce(max_len, op=torch.distributed.ReduceOp.MAX)
        pad = int(max_len.item()) - doc_embeddings.size(1)
        if pad > 0:
            shape = list(doc_embeddings.shape)
            shape[1] = pad
            padding = torch.zeros(*shape, device=doc_embeddings.device, dtype=doc_embeddings.dtype)
            doc_embeddings = torch.cat((padding, doc_embeddings), dim=1)

    gathered = torch.cat(all_gather(doc_embeddings), dim=0)
    offset = torch.distributed.get_rank() * local_batch_size
    return gathered, offset


class WithGradCache(nn.Module):
    """
    GradCache wrapper for contrastive losses (Gao et al., 2021).

    Wraps any bi-encoder or late-interaction loss (e.g. ``BiEncoderLoss``, ``BiNegativeCELoss``,
    ``ColbertLoss``, ``ColbertNegativeCELoss``) to enable large effective batch sizes under a
    fixed memory budget. Embeddings are computed in mini-batches without retaining the full
    activation graph; per-embedding gradients are cached from the loss and replayed through a
    backward hook that recomputes each mini-batch with gradients enabled.

    All scoring and loss math is delegated to ``loss``, so this wrapper stays compatible with
    every pooled or token-level contrastive loss and adds no loss-specific logic.

    Args:
        loss: The contrastive loss module to wrap. Its ``forward`` must accept
            ``query_embeddings``, ``doc_embeddings``, an ``offset`` keyword, and optionally
            ``neg_doc_embeddings``.
        mini_batch_size: Number of items embedded per mini-batch.
        show_progress_bar: Show a progress bar while embedding mini-batches.
    """

    def __init__(self, loss: nn.Module, mini_batch_size: int = 32, show_progress_bar: bool = False):
        super().__init__()
        self.loss = loss
        self.mini_batch_size = mini_batch_size
        self.show_progress_bar = show_progress_bar
        # Read by ContrastiveTrainer to route through the GradCache code path.
        self.gradcache_enabled = True
        # Toggled by ContrastiveTrainer; only gathers when running distributed.
        self.gather_across_processes = True

    @staticmethod
    def _autocast_ctx(device_type: str):
        """Replay the ambient autocast policy so both forward passes use the same dtype."""
        try:
            enabled = torch.is_autocast_enabled(device_type)
            dtype = torch.get_autocast_dtype(device_type)
        except TypeError:
            # torch < 2.4 has no device-type argument; fall back to the per-device queries.
            if device_type == "cpu":
                enabled, dtype = torch.is_autocast_cpu_enabled(), torch.get_autocast_cpu_dtype()
            else:
                enabled, dtype = torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()
        if enabled:
            return partial(torch.autocast, device_type=device_type, dtype=dtype)
        return contextlib.nullcontext

    def _embed_in_minibatches(
        self, model: nn.Module, features: Dict[str, torch.Tensor], autocast
    ) -> Tuple[List[torch.Tensor], List[RandContext]]:
        bsz = features["input_ids"].size(0)
        reps: List[torch.Tensor] = []
        rand_states: List[RandContext] = []
        for start in tqdm.trange(
            0, bsz, self.mini_batch_size, desc="Embedding minibatches", disable=not self.show_progress_bar
        ):
            mini = {k: v[start : start + self.mini_batch_size] for k, v in features.items()}
            rand_states.append(RandContext(*mini.values()))
            with torch.no_grad(), autocast():
                embeds = model(**mini)
            reps.append(embeds.detach().requires_grad_(True))
        return reps, rand_states

    def _compute_loss(self, reps: List[List[torch.Tensor]], num_neg_docs: int, with_backward: bool) -> torch.Tensor:
        query_embeddings = torch.cat(reps[0], dim=0)
        doc_embeddings = torch.cat(reps[1], dim=0)
        gathered_doc, offset = _gather_doc_embeddings(
            doc_embeddings, query_embeddings.size(0) if self.gather_across_processes else 0
        )

        kwargs = {"query_embeddings": query_embeddings, "doc_embeddings": gathered_doc, "offset": offset}
        if num_neg_docs:
            neg_embeddings = torch.cat(reps[2], dim=0)
            neg_embeddings = neg_embeddings.reshape(-1, num_neg_docs, *neg_embeddings.shape[1:])
            kwargs["neg_doc_embeddings"] = neg_embeddings

        loss = self.loss(**kwargs)
        if with_backward:
            loss.backward()
        return loss

    def _backward_hook(self, grad_output, model, branches, rand_states, cache, autocast):
        with torch.enable_grad():
            for features, branch_cache, branch_states in zip(branches, cache, rand_states):
                bsz = features["input_ids"].size(0)
                for i, start in enumerate(range(0, bsz, self.mini_batch_size)):
                    mini = {k: v[start : start + self.mini_batch_size] for k, v in features.items()}
                    with branch_states[i], autocast():
                        embeds = model(**mini)
                    # Replay the cached gradient: d(surrogate)/d(params) == d(loss)/d(params).
                    surrogate = torch.dot(embeds.flatten(), branch_cache[i].flatten()) * grad_output
                    surrogate.backward()

    def forward(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        query_prefix: str = "query_",
        pos_doc_prefix: str = "doc_",
        neg_doc_prefix: str = "neg_doc_",
    ) -> torch.Tensor:
        query_features = {k[len(query_prefix) :]: v for k, v in inputs.items() if k.startswith(query_prefix)}
        doc_features = {k[len(pos_doc_prefix) :]: v for k, v in inputs.items() if k.startswith(pos_doc_prefix)}
        neg_features = {k[len(neg_doc_prefix) :]: v for k, v in inputs.items() if k.startswith(neg_doc_prefix)}

        # Flatten negatives from (batch, num_negs, ...) to (batch * num_negs, ...) for embedding.
        num_neg_docs = 0
        if neg_features:
            num_neg_docs = neg_features["input_ids"].size(1)
            neg_features = {k: v.reshape(-1, *v.shape[2:]) for k, v in neg_features.items()}

        branches = [query_features, doc_features]
        if num_neg_docs:
            branches.append(neg_features)

        # First pass: embed every branch in mini-batches without gradients.
        # Capture the ambient autocast policy so the backward-hook re-forward (which runs
        # outside autocast during backprop) reproduces the same dtype, keeping grads exact.
        autocast = self._autocast_ctx(next(iter(inputs.values())).device.type)
        reps: List[List[torch.Tensor]] = []
        rand_states: List[List[RandContext]] = []
        for features in branches:
            branch_reps, branch_states = self._embed_in_minibatches(model, features, autocast)
            reps.append(branch_reps)
            rand_states.append(branch_states)

        if not torch.is_grad_enabled():
            return self._compute_loss(reps, num_neg_docs, with_backward=False)

        # Build the gradient cache, then replay it through a backward hook.
        loss = self._compute_loss(reps, num_neg_docs, with_backward=True)
        cache = [[mini.grad for mini in branch] for branch in reps]
        loss = loss.detach().requires_grad_()
        loss.register_hook(
            partial(
                self._backward_hook,
                model=model,
                branches=branches,
                rand_states=rand_states,
                cache=cache,
                autocast=autocast,
            )
        )
        return loss
