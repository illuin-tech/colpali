# from mteb.evaluation.evaluators.RetrievalEvaluator
from __future__ import annotations

import logging
from typing import Dict, Union

from datasets import Dataset
from peft import PeftModel
from transformers import PreTrainedModel, TrainerControl, TrainerState, TrainingArguments
from transformers.integrations import WandbCallback
from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorBEIR, ViDoReEvaluatorQA
from vidore_benchmark.retrievers import VisionRetriever

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

logger = logging.getLogger(__name__)


METRICS_TO_TRACK = [
    "ndcg_at_1",
    "ndcg_at_3",
    "ndcg_at_5",
    "ndcg_at_10",
    "ndcg_at_50",
    "ndcg_at_100",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "recall_at_50",
    "recall_at_100",
    "map_at_1",
    "map_at_3",
    "map_at_5",
    "map_at_10",
    "map_at_50",
    "map_at_100",
]


class BenchmarkEvalCallback(WandbCallback):
    def __init__(
        self,
        processor,
        model,
        eval_dataset_loader,
        batch_query: int = 4,
        batch_passage: int = 4,
        batch_score: int = 4,
        run_frequency: int = 5,
        dataset_format: str = "beir",
    ):
        """
        Callback to evaluate the model on a collection of datasets during training.

        Args:
            processor: The processor to use for the model.
            model: The model to evaluate.
            eval_dataset_loader: A dictionary of dataset loading functions.
            batch_query: Batch size for query.
            batch_passage: Batch size for passage.
            batch_score: Batch size for scoring.
            run_frequency: Frequency of evaluation in steps.
            dataset_format: Format of the evaluation dataset, either "beir" or "qa".
        """
        self.processor = processor
        self.model = model
        self.eval_dataset_loader = eval_dataset_loader
        self.batch_query = batch_query
        self.batch_passage = batch_passage
        self.batch_score = batch_score
        self.eval_steps_frequency = run_frequency
        self.counter_eval = 0
        self.eval_dataset_format = dataset_format
        super().__init__()

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.eval_steps_frequency != 0:
            self.counter_eval += 1
            return
        else:
            self.counter_eval = 1

        if self.processor is None:
            print("Processor not provided. Skipping benchmark evaluation.")
            return

        print(f"\n=== Running benchmark evaluation at global step {state.global_step} ===")

        # Evaluate on a collection.
        if self.eval_dataset_loader is not None:
            try:
                metrics_collection = {}
                for test_name, test_dataset_loading_func in self.eval_dataset_loader.items():
                    metrics = evaluate_dataset(
                        model=self.model,
                        processor=self.processor,
                        dataset=test_dataset_loading_func(),
                        format=self.eval_dataset_format,
                        batch_passage=self.batch_passage,
                        batch_query=self.batch_query,
                        batch_score=self.batch_score,
                    )
                    metrics_collection[test_name] = {k: v for k, v in metrics.items() if k in METRICS_TO_TRACK}
                print(f"Benchmark metrics for tests datasets at step {state.global_step}:")
                print(metrics_collection)
                print("logging metrics to wandb")
                self._wandb.log(metrics_collection)
            except Exception as e:
                print(f"Error during benchmark evaluation on collection '{self.eval_collection}': {e}")

        # Set model back to train mode.
        self.model.train()
        return


def evaluate_dataset(
    model: Union[PreTrainedModel, PeftModel],
    processor: BaseVisualRetrieverProcessor,
    dataset: Dataset,
    format: str = "beir",
    batch_passage: int = 4,
    batch_query: int = 4,
    batch_score: int = 4,
) -> Dict[str, float]:
    """
    Evaluate a dataset using the vidore-benchmark library.
    """
    model.eval()

    # Create a vision retriever with the current model checkpoint.
    vision_retriever = VisionRetriever(
        model=model,
        processor=processor,
    )

    if format == "qa":
        vidore_evaluator = ViDoReEvaluatorQA(vision_retriever)
    elif format == "beir":
        vidore_evaluator = ViDoReEvaluatorBEIR(vision_retriever)
    else:
        raise ValueError(f"Invalid dataset format: {format}, must be 'qa' or 'beir'")

    metrics = vidore_evaluator.evaluate_dataset(
        ds=dataset,
        batch_query=batch_query,
        batch_passage=batch_passage,
        batch_score=batch_score,
    )

    return metrics
