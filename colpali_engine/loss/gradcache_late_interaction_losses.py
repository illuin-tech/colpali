import torch.nn as nn
from functools import partial
import tqdm


from torch.utils.checkpoint import get_device_states, set_device_states
import torch


class RandContext:
    """
    Random-state context manager that captures both CPU and GPU random states.
    This ensures that when re‑executing a forward pass (e.g. in GradCache’s second pass),
    stochastic operations produce identical outputs.
    """
    def __init__(self, *tensors) -> None:
        # Capture CPU RNG state.
        self.fwd_cpu_state = torch.get_rng_state()
        # Capture GPU states for all devices associated with the provided tensors.
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        # Fork the RNG states on the captured devices.
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        # Reset the CPU RNG state.
        torch.set_rng_state(self.fwd_cpu_state)
        # Reset the GPU RNG states.
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None



def _backward_hook(grad_output, sentence_features, random_states, loss_obj, model):
    """
    Backward hook that re-computes the embeddings in mini-batches with gradients enabled
    and uses the cached gradients to backpropagate. This version wraps the forward pass in the
    corresponding RandContext to reproduce the same randomness.
    """
    mini_batch_size = loss_obj.mini_batch_size
    # sentence_features: a list with two dicts [query_features, doc_features]
    # random_states: a list with two lists of RandContext objects.
    for branch_feature, branch_cache, branch_random_states in zip(sentence_features, random_states):
        input_ids = branch_feature["input_ids"]
        bsz = input_ids.size(0)
        # Iterate over mini-batches.
        for idx, start in enumerate(range(0, bsz, mini_batch_size)):
            end = start + mini_batch_size
            mini_feature = {k: v[start:end] for k, v in branch_feature.items()}
            # Use the stored RandContext if available.
            r_state = branch_random_states[idx]
            if r_state is not None:
                with r_state:
                    mini_embeds = model.inner_forward(**mini_feature)
            else:
                mini_embeds = model.inner_forward(**mini_feature)
            mini_embeds = mini_embeds.detach().requires_grad_(True)
            cached_grad = branch_cache[idx]
            # Compute a surrogate loss that replays the cached gradient.
            surrogate = torch.dot(mini_embeds.flatten(), cached_grad.flatten()) * grad_output
            surrogate.backward()


class GradCacheColbertLoss(nn.Module):
    def __init__(self, mini_batch_size: int = 32, scale: float = 1.0, show_progress_bar: bool = False):
        """
        GradCache enabled version of the ColBERT loss.

        Args:
            mini_batch_size: Number of items per mini-batch.
            scale: Scaling factor for the similarity scores.
            show_progress_bar: If True, shows progress bars during mini-batch processing.
        """
        super().__init__()
        self.mini_batch_size = mini_batch_size
        self.scale = scale
        self.ce_loss = nn.CrossEntropyLoss()
        self.cache = None
        self.random_states = None
        self.show_progress_bar = show_progress_bar
        self.gradcache_enabled = True  # Flag indicating GradCache is active.

    def embed_minibatch_iter(self, model, sentence_feature: dict, with_grad: bool, copy_random_state: bool):
        input_ids = sentence_feature["input_ids"]
        bsz = input_ids.size(0)
        for start in tqdm.trange(0, bsz, self.mini_batch_size, desc="Embedding minibatches",
                                 disable=not self.show_progress_bar):
            end = start + self.mini_batch_size
            mini_feature = {k: v[start:end] for k, v in sentence_feature.items()}
            random_state = None
            if copy_random_state:
                random_state = RandContext(*mini_feature.values())
            grad_context = torch.enable_grad() if with_grad else torch.no_grad()
            with grad_context:
                breakpoint()
                mini_embeds = model.inner_forward(**mini_feature)
                mini_embeds = mini_embeds.detach().requires_grad_(with_grad)
            yield mini_embeds, random_state

    def calculate_loss(self, reps: list[list[torch.Tensor]], with_backward: bool = False) -> torch.Tensor:
        """
        Calculate the ColBERT-style loss.
        reps: list with two elements – reps[0] for query embeddings, reps[1] for doc embeddings.
        Each element is a list of mini-batch tensors.
        """
        embeddings_query = torch.cat(reps[0], dim=0)  # shape: (total_query, seq_len, dim)
        embeddings_doc = torch.cat(reps[1], dim=0)  # shape: (total_doc, seq_len, dim)
        scores = torch.einsum("bnd,csd->bcns", embeddings_query, embeddings_doc).max(dim=3)[0].sum(dim=2)
        batch_size = scores.size(0)
        labels = torch.arange(batch_size, device=scores.device)
        loss = self.ce_loss(scores * self.scale, labels)
        if with_backward:
            loss.backward()
        return loss

    def calculate_loss_and_cache_gradients(self, reps: list[list[torch.Tensor]]) -> torch.Tensor:
        loss = self.calculate_loss(reps, with_backward=True)
        loss = loss.detach().requires_grad_()
        # Cache gradients for each mini-batch.
        self.cache = []
        for branch in reps:
            branch_cache = []
            for r in branch:
                branch_cache.append(r.grad)
            self.cache.append(branch_cache)
        return loss

    def forward(self, model, inputs: dict) -> torch.Tensor:
        """
        inputs: dict containing keys with prefixes "query_" and "doc_".
        """
        # Remove prefixes.
        query_features = {k.replace("query_", ""): v for k, v in inputs.items() if k.startswith("query_")}
        doc_features = {k.replace("doc_", ""): v for k, v in inputs.items() if k.startswith("doc_")}

        # === First Pass: Get embeddings without gradients, capturing RandContext.
        reps_query = []
        rs_query = []
        for mini_embeds, rs in self.embed_minibatch_iter(model, query_features, with_grad=False,
                                                         copy_random_state=True):
            reps_query.append(mini_embeds)
            rs_query.append(rs)
        reps_doc = []
        rs_doc = []
        for mini_embeds, rs in self.embed_minibatch_iter(model, doc_features, with_grad=False, copy_random_state=True):
            reps_doc.append(mini_embeds)
            rs_doc.append(rs)
        reps = [reps_query, reps_doc]
        self.random_states = [rs_query, rs_doc]

        if torch.is_grad_enabled():
            # Step (2): Compute loss and cache gradients.
            loss = self.calculate_loss_and_cache_gradients(reps)
            # Step (3): Re-run embeddings with gradients enabled and register a backward hook that uses the cached gradients.
            loss.register_hook(partial(_backward_hook, sentence_features=[query_features, doc_features],
                                       random_states=self.random_states, loss_obj=self, model=model))
        else:
            loss = self.calculate_loss(reps, with_backward=False)
        return loss
