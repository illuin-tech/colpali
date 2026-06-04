"""Run a few training steps and record the VRAM used by every MaxSim call.

``maxsim_inbatch`` is wrapped at the loss-module level so each call records:
- ``forward_transient_peak_mib``: extra memory while the op's forward runs, freed after.
- ``saved_for_backward_mib``: memory held from forward until backward (vanilla keeps the
  ``[B, B, Lq, Ld]`` score tensor; LIK keeps the ``[B, B]`` output).
If the op itself OOMs, that is recorded too — it pins the OOM inside the score
computation rather than the model.

The op's backward cannot be bracketed in-train: a tensor grad hook is a pre-hook on the
*producing* node, so a "close bracket" on ``query`` only fires when the query tower's
backward is scheduled — after the whole doc-tower backward ran. Instead, each recorded
(shape, dtype) is replayed after training on fresh random embeddings whose graph contains
only the op, where forward/saved/backward peaks bracket exactly; the replayed forward
numbers double as a fidelity check against the in-train ones.

``maxsim_kd`` is not instrumented: only the negative-doc losses call it and the bench
config uses ``ColbertPairwiseCELoss`` (in-batch only).

Usage:
    COLPALI_SCORES_BACKEND=torch python bench_lik/bench_train.py \\
        --config bench_lik/bench_config_subset.yaml \\
        --batch-size 64 --max-steps 4 \\
        --output bench_lik/results/maxsim_vram_b64_torch.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable

import configue
import torch
from transformers import TrainerCallback

from colpali_engine.loss import late_interaction_losses
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils._lik_backend import is_available as lik_is_available

MaxsimFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class MaxsimVramRecorder:
    """Per-call VRAM for the maxsim op, plus a run-level peak that survives the per-op stat resets."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.run_peak_alloc_bytes: int = 0
        self.run_peak_reserved_bytes: int = 0

    def fold_run_peak(self) -> None:
        """Capture the global peak before a reset wipes it; call once more after training."""
        self.run_peak_alloc_bytes = max(self.run_peak_alloc_bytes, torch.cuda.max_memory_allocated())
        self.run_peak_reserved_bytes = max(self.run_peak_reserved_bytes, torch.cuda.max_memory_reserved())

    def wrap(self, fn: MaxsimFn, op_name: str) -> MaxsimFn:
        def wrapped(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
            torch.cuda.synchronize()
            self.fold_run_peak()
            before_forward_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            record: dict = {
                "op": op_name,
                "query_shape": list(query.shape),
                "doc_shape": list(doc.shape),
                "dtype": str(query.dtype),
            }
            try:
                out = fn(query, doc)
            except torch.cuda.OutOfMemoryError:
                record["oom_in_forward"] = True
                self.calls.append(record)
                raise
            torch.cuda.synchronize()
            record["forward_transient_peak_mib"] = (torch.cuda.max_memory_allocated() - before_forward_bytes) / 2**20
            record["saved_for_backward_mib"] = (torch.cuda.memory_allocated() - before_forward_bytes) / 2**20
            self.calls.append(record)
            return out

        return wrapped


class StepTimerCallback(TrainerCallback):
    """Record wall time for each optimizer step."""

    def __init__(self) -> None:
        self.step_times: list[float] = []
        self._step_start: float | None = None

    def on_step_begin(self, args, state, control, **kwargs) -> None:
        torch.cuda.synchronize()
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs) -> None:
        torch.cuda.synchronize()
        if self._step_start is not None:
            self.step_times.append(time.perf_counter() - self._step_start)
            self._step_start = None


def _replay_op_isolated(fn: MaxsimFn, record: dict) -> dict:
    """Replay one recorded call on fresh random embeddings whose graph contains only the op,
    so the forward/saved/backward peaks bracket exactly. Returns MiB deltas, or an OOM marker."""
    dtype = getattr(torch, record["dtype"].removeprefix("torch."))
    query = torch.randn(record["query_shape"], dtype=dtype, device="cuda", requires_grad=True)
    doc = torch.randn(record["doc_shape"], dtype=dtype, device="cuda", requires_grad=True)

    torch.cuda.synchronize()
    before_forward_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        out = fn(query, doc)
        torch.cuda.synchronize()
        forward_peak_mib = (torch.cuda.max_memory_allocated() - before_forward_bytes) / 2**20
        saved_mib = (torch.cuda.memory_allocated() - before_forward_bytes) / 2**20

        before_backward_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out.sum().backward()
        torch.cuda.synchronize()
        backward_peak_mib = (torch.cuda.max_memory_allocated() - before_backward_bytes) / 2**20
    except torch.cuda.OutOfMemoryError:
        return {**record, "oom_in_replay": True}

    return {
        **record,
        "forward_transient_peak_mib": forward_peak_mib,
        "saved_for_backward_mib": saved_mib,
        "backward_transient_peak_mib": backward_peak_mib,
    }


def _lik_version() -> str:
    if not lik_is_available():
        return "not-available"
    import late_interaction_kernels as lik

    return getattr(lik, "__version__", "unknown")


def _summarize_calls(calls: list[dict]) -> dict:
    """Max per-call numbers — memory is shape-deterministic, so max is the story."""
    metric_names = ["forward_transient_peak_mib", "saved_for_backward_mib", "backward_transient_peak_mib"]
    summary: dict = {
        "num_calls": len(calls),
        "num_oom": sum(1 for call in calls if call.get("oom_in_forward") or call.get("oom_in_replay")),
    }
    for metric in metric_names:
        summary[f"max_{metric}"] = max((call[metric] for call in calls if metric in call), default=None)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to bench training config yaml.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the JSON metrics.")
    parser.add_argument("--max-steps", type=int, default=4, help="Training steps; memory peaks settle by step 1.")
    parser.add_argument("--batch-size", type=int, default=0, help="If >0, override per_device_train_batch_size.")
    args = parser.parse_args()

    backend = os.environ.get("COLPALI_SCORES_BACKEND", "auto")
    print(f"COLPALI_SCORES_BACKEND={backend} · LIK available: {lik_is_available()} ({_lik_version()})")

    config = configue.load(args.config, sub_path="config")
    if not isinstance(config, ColModelTrainingConfig):
        raise ValueError("Config must be of type ColModelTrainingConfig")
    config.tr_args.max_steps = args.max_steps
    if args.batch_size > 0:
        config.tr_args.per_device_train_batch_size = args.batch_size

    recorder = MaxsimVramRecorder()
    # The losses bound the dispatcher at import time, so patch the loss module's reference.
    unwrapped_maxsim_inbatch = late_interaction_losses.maxsim_inbatch
    late_interaction_losses.maxsim_inbatch = recorder.wrap(unwrapped_maxsim_inbatch, "maxsim_inbatch")

    training_app = ColModelTraining(config)

    # Inline ColModelTraining.train() so the timer callback can be attached.
    from colpali_engine.collators import VisualRetrieverCollator
    from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer

    collator = VisualRetrieverCollator(
        processor=training_app.config.processor,
        max_length=training_app.config.max_length,
    )
    timer_cb = StepTimerCallback()
    trainer = ContrastiveTrainer(
        model=training_app.model,
        train_dataset=training_app.train_dataset,
        eval_dataset=None,
        args=training_app.config.tr_args,
        data_collator=collator,
        loss_func=training_app.config.loss_func,
        is_vision_model=training_app.config.processor is not None,
        callbacks=[timer_cb],
    )
    trainer.args.remove_unused_columns = False

    torch.cuda.reset_peak_memory_stats()
    base_payload: dict = {
        "colpali_scores_backend": backend,
        "lik_version": _lik_version(),
        "device_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "batch_size": training_app.config.tr_args.per_device_train_batch_size,
    }

    # OOM is an expected sweep outcome: record it and exit 0 so the driver keeps going.
    oom = False
    oom_message: str | None = None
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as error:
        # The first line names the failed allocation ("Tried to allocate X GiB") — which term crossed.
        oom = True
        oom_message = str(error).split("\n")[0]
    recorder.fold_run_peak()

    # Free what an OOMed run left behind so the isolated replays start from a clean allocator.
    torch.cuda.empty_cache()
    isolated_calls = [_replay_op_isolated(unwrapped_maxsim_inbatch, record) for record in recorder.calls]

    payload = {
        **base_payload,
        "oom": oom,
        "oom_message": oom_message,
        "step_peak_alloc_mib": recorder.run_peak_alloc_bytes / 2**20,
        "step_peak_reserved_mib": recorder.run_peak_reserved_bytes / 2**20,
        "step_times_sec": timer_cb.step_times,
        "maxsim_in_train_summary": _summarize_calls(recorder.calls),
        "maxsim_isolated_summary": _summarize_calls(isolated_calls),
        "maxsim_calls_in_train": recorder.calls,
        "maxsim_calls_isolated": isolated_calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
