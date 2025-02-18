from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from collections.abc import Iterator
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import get_device_states, set_device_states

# ------------------------------------------------------------------------------
# Utility: A context manager that saves/restores RNG state.
# ------------------------------------------------------------------------------
class RandContext:
    def __init__(self, *tensors) -> None:
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

# ------------------------------------------------------------------------------
# Backward hook for multi–input grad cache.
# ------------------------------------------------------------------------------
def _backward_hook_multi(grad_output: Tensor, features_dict: Dict[str, dict], loss_obj: CachedColbertLossBase) -> None:
    """
    For each input branch (e.g. 'query', 'doc', 'neg_doc'), re–run the forward pass
    (with gradients) using the saved RNG contexts and then backpropagate the cached gradients.
    """
    with torch.enable_grad():
        for key, feat in features_dict.items():
            cached_grads = loss_obj.cache[key]  # list (one per mini–batch)
            rand_states = loss_obj.random_states[key]
            for (reps_mb, _), grad_mb, rand_state in zip(
                loss_obj.embed_minibatch_iter(feat, with_grad=True, copy_random_state=False, random_states=rand_states),
                cached_grads,
                rand_states
            ):
                surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                surrogate.backward()

# ------------------------------------------------------------------------------
# Base class that implements grad cache embedding passes.
# ------------------------------------------------------------------------------
class CachedColbertLossBase(nn.Module):
    def __init__(self, model: nn.Module, mini_batch_size: int = 32, show_progress_bar: bool = False) -> None:
        """
        model: a SentenceTransformer–like model which, given a features dict,
               returns a dict with key "sentence_embedding" of shape (bsz, num_tokens, dim)
        """
        super().__init__()
        self.model = model
        self.mini_batch_size = mini_batch_size
        self.show_progress_bar = show_progress_bar
        # These will be dictionaries keyed by input type (e.g. 'query', 'doc', etc.)
        self.cache: Dict[str, list[Tensor]] = {}
        self.random_states: Dict[str, list[RandContext]] = {}

    def embed_minibatch(
        self,
        features: dict,
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """
        Run the model on a mini–batch of the features.
        """
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        features_mb = {k: v[begin:end] for k, v in features.items()}
        with random_state_context:
            with grad_context():
                new_rand_state = RandContext(*features_mb.values()) if copy_random_state else None
                # Expect model(features) returns a dict with key "sentence_embedding"
                reps = self.model(features_mb)["sentence_embedding"]
        return reps, new_rand_state

    def embed_minibatch_iter(
        self,
        features: dict,
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        input_ids: Tensor = features["input_ids"]
        bsz = input_ids.shape[0]
        for i in range(0, bsz, self.mini_batch_size):
            e = i + self.mini_batch_size
            reps, new_rand_state = self.embed_minibatch(
                features=features,
                begin=i,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, new_rand_state

    def _embed_all(self, features: dict) -> tuple[list[Tensor], list[RandContext]]:
        reps_list = []
        rand_state_list = []
        for reps_mb, rand_state in self.embed_minibatch_iter(features, with_grad=False, copy_random_state=True):
            # Detach and mark for gradient in the second pass.
            reps_list.append(reps_mb.detach().requires_grad_())
            rand_state_list.append(rand_state)
        return reps_list, rand_state_list

    def _aggregate_embeddings(self, reps: list[Tensor]) -> Tensor:
        return torch.cat(reps, dim=0)

    def forward(self, **features: dict) -> Tensor:
        """
        Expects keyword–arguments for each input branch.
        For example:
            forward(query=..., doc=...)
            or forward(query=..., doc=..., neg_doc=...)
        Each input is a features dict (with keys like "input_ids").
        """
        reps: dict[str, list[Tensor]] = {}
        rand_states: dict[str, list[RandContext]] = {}
        for key, feat in features.items():
            reps[key], rand_states[key] = self._embed_all(feat)
        self.random_states = rand_states

        if torch.is_grad_enabled():
            loss = self._compute_loss_and_cache_gradients(**reps)
            loss.register_hook(partial(_backward_hook_multi, features_dict=features, loss_obj=self))
        else:
            agg = {key: self._aggregate_embeddings(reps[key]) for key in reps}
            if "neg_doc" in agg:
                loss = self._compute_loss(agg["query"], agg["doc"], agg["neg_doc"], with_backward=False)
            else:
                loss = self._compute_loss(agg["query"], agg["doc"], with_backward=False)
        return loss

    # The following two methods are meant to be implemented by subclasses:
    def _compute_loss(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def _compute_loss_and_cache_gradients(self, **reps: list[Tensor]) -> Tensor:
        # In our subclasses we first aggregate the mini–batches and then compute the loss in a mini–batch loop.
        raise NotImplementedError

# ------------------------------------------------------------------------------
# Cached ColBERT Loss (simple cross–entropy over scores)
# ------------------------------------------------------------------------------
class CachedColbertLoss(CachedColbertLossBase):
    def __init__(self, model: nn.Module, mini_batch_size: int = 32, show_progress_bar: bool = False) -> None:
        super().__init__(model, mini_batch_size, show_progress_bar)
        self.ce_loss = CrossEntropyLoss()

    def _compute_loss(self, query: Tensor, doc: Tensor, with_backward: bool = False) -> Tensor:
        # query: (B, Nq, D), doc: (B, Nd, D)
        batch_size = query.shape[0]
        # Compute scores:
        #   scores: (B, B, Nq, Nd)
        scores = torch.einsum("bnd,csd->bcns", query, doc)
        # For each query-document pair, take the max over document tokens then sum over query tokens.
        scores = scores.max(dim=3)[0].sum(dim=2)  # shape: (B, B)
        labels = torch.arange(batch_size, device=query.device)
        loss_total = 0.0
        for i in range(0, batch_size, self.mini_batch_size):
            j = i + self.mini_batch_size
            scores_mbatch = scores[i:j]  # (mb, B)
            loss_mbatch = self.ce_loss(scores_mbatch, labels[i:j])
            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            loss_total = loss_total + loss_mbatch * (scores_mbatch.shape[0] / batch_size)
        return loss_total

    def _compute_loss_and_cache_gradients(self, **reps: list[Tensor]) -> Tensor:
        agg_query = self._aggregate_embeddings(reps["query"])
        agg_doc = self._aggregate_embeddings(reps["doc"])
        loss = self._compute_loss(agg_query, agg_doc, with_backward=True)
        return loss.detach().requires_grad_()

# ------------------------------------------------------------------------------
# Cached ColBERT Pairwise CE Loss
# ------------------------------------------------------------------------------
class CachedColbertPairwiseCELoss(CachedColbertLossBase):
    def __init__(self, model: nn.Module, mini_batch_size: int = 32, show_progress_bar: bool = False) -> None:
        super().__init__(model, mini_batch_size, show_progress_bar)
        self.ce_loss = CrossEntropyLoss()

    def _compute_loss(self, query: Tensor, doc: Tensor, with_backward: bool = False) -> Tensor:
        batch_size = query.shape[0]
        scores = torch.einsum("bnd,csd->bcns", query, doc)
        scores = scores.max(dim=3)[0].sum(dim=2)  # (B, B)
        pos_scores = scores.diagonal()  # (B,)
        mask = torch.eye(batch_size, device=scores.device) * 1e6
        neg_scores = (scores - mask).max(dim=1)[0]
        loss_total = F.softplus(neg_scores - pos_scores).mean()
        if with_backward:
            loss_total.backward()
            loss_total = loss_total.detach()
        return loss_total

    def _compute_loss_and_cache_gradients(self, **reps: list[Tensor]) -> Tensor:
        agg_query = self._aggregate_embeddings(reps["query"])
        agg_doc = self._aggregate_embeddings(reps["doc"])
        loss = self._compute_loss(agg_query, agg_doc, with_backward=True)
        return loss.detach().requires_grad_()

# ------------------------------------------------------------------------------
# Cached ColBERT Pairwise Negative CE Loss
# ------------------------------------------------------------------------------
class CachedColbertPairwiseNegativeCELoss(CachedColbertLossBase):
    def __init__(self, model: nn.Module, in_batch_term: bool = False, mini_batch_size: int = 32, show_progress_bar: bool = False) -> None:
        super().__init__(model, mini_batch_size, show_progress_bar)
        self.in_batch_term = in_batch_term

    def _compute_loss(self, query: Tensor, doc: Tensor, neg_doc: Tensor, with_backward: bool = False) -> Tensor:
        # Compute positive and negative scores using token-level max and sum.
        pos_scores = torch.einsum("bnd,bsd->bns", query, doc).max(dim=2)[0].sum(dim=1)
        neg_scores = torch.einsum("bnd,bsd->bns", query, neg_doc).max(dim=2)[0].sum(dim=1)
        loss_total = F.softplus(neg_scores - pos_scores).mean()
        if self.in_batch_term:
            scores = torch.einsum("bnd,csd->bcns", query, doc)
            scores = scores.max(dim=3)[0].sum(dim=2)
            pos_scores_ib = scores.diagonal()
            mask = torch.eye(scores.shape[0], device=scores.device) * 1e6
            neg_scores_ib = (scores - mask).max(dim=1)[0]
            loss_total = loss_total + F.softplus(neg_scores_ib - pos_scores_ib).mean()
            loss_total = loss_total / 2
        if with_backward:
            loss_total.backward()
            loss_total = loss_total.detach()
        return loss_total

    def _compute_loss_and_cache_gradients(self, **reps: list[Tensor]) -> Tensor:
        agg_query = self._aggregate_embeddings(reps["query"])
        agg_doc = self._aggregate_embeddings(reps["doc"])
        agg_neg_doc = self._aggregate_embeddings(reps["neg_doc"])
        loss = self._compute_loss(agg_query, agg_doc, agg_neg_doc, with_backward=True)
        return loss.detach().requires_grad_()
