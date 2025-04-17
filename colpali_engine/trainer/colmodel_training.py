import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    PreTrainedModel,
    TrainingArguments,
)

from colpali_engine.collators import CorpusQueryCollator, VisualRetrieverCollator
from colpali_engine.loss.late_interaction_losses import (
    ColbertLoss,
)
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.trainer.eval_utils import BenchmarkEvalCallback, evaluate_dataset
from colpali_engine.utils.gpu_stats import print_gpu_utilization, print_summary
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


@dataclass
class ColModelTrainingConfig:
    """Configuration for training a ColVision model.

    Args:
        model (Union[PreTrainedModel, PeftModel]): Base model to train.
        processor (BaseVisualRetrieverProcessor): Processor for visual data processing.
        tr_args (Optional[TrainingArguments]): Transformers training arguments. If not provided, uses default values.
        output_dir (Optional[str]): Output directory to save the model.
            If not provided, creates a path based on model name.
        max_length (int): Maximum sequence length for inputs. Default: 256.
        run_eval (bool): If True, runs evaluation. Default: True.
        run_train (bool): If True, runs training. Default: True.
        vidore_eval_frequency (int): Vidore evaluation frequency, must be a multiple of tr_args.eval_steps.
            Pass -1 to disable. Default: -1.
        eval_dataset_format (str): Evaluation dataset format ("qa" or "beir"). Default: "qa".
        peft_config (Optional[LoraConfig]): PEFT configuration if used. Default: None.
        loss_func (Optional[Callable]): Custom loss function. Default: ColbertLoss().
        dataset_loading_func (Optional[Callable]): Dataset loading function. Default: None.
        eval_dataset_loader (Optional[Dict[str, Callable]]): Evaluation dataset loaders. Default: None.
        pretrained_peft_model_name_or_path (Optional[str]): Path to a pretrained PEFT model. Default: None.
    """

    model: Union[PreTrainedModel, PeftModel]
    processor: BaseVisualRetrieverProcessor
    tr_args: Optional[TrainingArguments] = None
    output_dir: Optional[str] = None
    max_length: int = 256
    run_eval: bool = True
    run_train: bool = True
    vidore_eval_frequency: int = -1
    eval_dataset_format: str = "qa"
    peft_config: Optional[LoraConfig] = None
    loss_func: Optional[Callable] = ColbertLoss()
    dataset_loading_func: Optional[Callable] = None
    eval_dataset_loader: Optional[Dict[str, Callable]] = None
    pretrained_peft_model_name_or_path: Optional[str] = None
    """
    Config class used for training a ColVision model.
    """

    def __post_init__(self):
        """
        Initialize the model and tokenizer if not provided
        """
        if self.output_dir is None:
            sanitized_name = str(self.model.name_or_path).replace("/", "_")
            self.output_dir = f"./models/{sanitized_name}"

        if self.tr_args is None:
            print("No training arguments provided. Using default.")
            self.tr_args = TrainingArguments(output_dir=self.output_dir)
        elif self.tr_args.output_dir is None or self.tr_args.output_dir == "trainer_output":
            self.tr_args.output_dir = self.output_dir

        if isinstance(self.tr_args.learning_rate, str):
            print("Casting learning rate to float")
            self.tr_args.learning_rate = float(self.tr_args.learning_rate)

        self.tr_args.remove_unused_columns = False

        if self.pretrained_peft_model_name_or_path is not None:
            print("Loading pretrained PEFT model")
            self.model.load_adapter(self.pretrained_peft_model_name_or_path, is_trainable=True)

        if self.peft_config is not None:
            print("Configurating PEFT model")
            if self.pretrained_peft_model_name_or_path is None:
                self.model = get_peft_model(self.model, self.peft_config)
                self.model.print_trainable_parameters()
            else:
                print(f"Adapter already loaded from {self.pretrained_peft_model_name_or_path}. Not overwriting.")

    print_gpu_utilization()


class ColModelTraining:
    """
    Class that contains the training and evaluation logic for a ColVision model.
    """

    def __init__(self, config: ColModelTrainingConfig) -> None:
        self.config = config
        self.model = self.config.model
        self.current_git_hash = os.popen("git rev-parse HEAD").read().strip()
        self.dataset = self.config.dataset_loading_func()

        if isinstance(self.dataset, Tuple):
            print("Dataset has BEIR/hard negatives format. Using CorpusQueryCollator.")
            corpus_format = self.dataset[2]
            neg_dataset = self.dataset[1]
            self.dataset = self.dataset[0]
            self.collator = CorpusQueryCollator(
                processor=self.config.processor,
                max_length=self.config.max_length,
                image_dataset=neg_dataset,
                mined_negatives=True,
                corpus_format=corpus_format,
            )
        else:
            print("Dataset has QA format. Using VisualRetrieverCollator.")
            self.collator = VisualRetrieverCollator(
                processor=self.config.processor,
                max_length=self.config.max_length,
            )

    def train(self) -> None:
        if isinstance(self.collator, CorpusQueryCollator) and self.collator.mined_negatives:
            print("Training with hard negatives")
        else:
            print("Training with in-batch negatives")

        trainer = ContrastiveTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            args=self.config.tr_args,
            data_collator=self.collator,
            loss_func=self.config.loss_func,
            is_vision_model=self.config.processor is not None,
        )

        trainer.args.remove_unused_columns = False

        if self.config.processor is not None and self.config.vidore_eval_frequency > 0:
            trainer.add_callback(
                BenchmarkEvalCallback(
                    processor=self.config.processor,
                    model=self.model,
                    eval_dataset_loader=self.config.eval_dataset_loader,
                    batch_query=self.config.tr_args.per_device_eval_batch_size,
                    batch_passage=4,
                    batch_score=4,
                    run_frequency=self.config.vidore_eval_frequency,
                    dataset_format=self.config.eval_dataset_format,
                )
            )

        result = trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint)
        print_summary(result)

    def eval(self) -> None:
        all_metrics = {}

        all_metrics["validation_set"] = evaluate_dataset(
            model=self.model,
            processor=self.config.processor,
            dataset=self.dataset["test"],
            format="qa",
            batch_passage=self.config.tr_args.per_device_eval_batch_size,
            batch_query=self.config.tr_args.per_device_eval_batch_size,
            batch_score=self.config.tr_args.per_device_eval_batch_size,
        )

        if self.config.eval_dataset_loader is not None:
            # Create a vision retriever with the current model checkpoint.
            eval_dataset_format = getattr(self.config.tr_args, "eval_dataset_format", "beir")

            for test_name, test_dataset_loading_func in self.config.eval_dataset_loader.items():
                print(f"Evaluating {test_name}")
                all_metrics[test_name] = evaluate_dataset(
                    model=self.model,
                    processor=self.config.processor,
                    dataset=test_dataset_loading_func(),
                    format=eval_dataset_format,
                    batch_passage=self.config.tr_args.per_device_eval_batch_size,
                    batch_query=self.config.tr_args.per_device_eval_batch_size,
                    batch_score=self.config.tr_args.per_device_eval_batch_size,
                )
                print(f"Metrics for {test_name}: {all_metrics[test_name]}")

    def save(self, config_file: str):
        # save model
        self.model.save_pretrained(self.config.output_dir)
        self.config.processor.save_pretrained(self.config.output_dir)

        # Copy-paste the training config
        os.system(f"cp {config_file} {self.config.output_dir}/training_config.yml")

        # Save git hash of the commit at beginning of training
        with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
            f.write(self.current_git_hash)
