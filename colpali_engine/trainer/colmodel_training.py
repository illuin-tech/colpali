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
from colpali_engine.trainer.eval_utils import BenchmarkEvalCallback
from colpali_engine.utils.gpu_stats import print_gpu_utilization, print_summary
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


@dataclass
class ColModelTrainingConfig:
    model: Union[PreTrainedModel, PeftModel]
    processor: BaseVisualRetrieverProcessor
    tr_args: Optional[TrainingArguments] = None
    output_dir: Optional[str] = None
    max_length: int = 256
    run_eval: bool = True
    run_train: bool = True
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

        if self.config.processor is not None and self.config.tr_args.get("run_vidore_evaluator", False):
            vidore_eval_dataset_name = self.config.tr_args.get("vidore_eval_dataset_name", None)
            vidore_eval_collection_name = self.config.tr_args.get("vidore_eval_collection_name", None)

            if vidore_eval_dataset_name is not None and vidore_eval_collection_name is not None:
                raise ValueError(
                    "Both vidore_eval_dataset_name and vidore_eval_collection_name are provided. "
                    "You should only provide one of the two"
                )
            elif vidore_eval_dataset_name is None and vidore_eval_collection_name is None:
                print("WARNING : No dataset provided for ViDoRe evaluation. Skipping evaluation.")
            else:
                trainer.add_callback(
                    BenchmarkEvalCallback(
                        processor=self.config.processor,
                        model=self.model,
                        eval_dataset_loader=self.config.eval_dataset_loader,
                        batch_query=self.config.tr_args.per_device_eval_batch_size,
                        batch_passage=4,
                        batch_score=4,
                        run_frequency=self.config.tr_args.get("eval_steps_frequency", 5),
                    )
                )

        result = trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint)
        print_summary(result)

    def save(self, config_file):
        # save model
        self.model.save_pretrained(self.config.output_dir)
        self.config.processor.save_pretrained(self.config.output_dir)

        # Copy-paste the training config
        os.system(f"cp {config_file} {self.config.output_dir}/training_config.yml")

        # Save git hash of the commit at beginning of training
        with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
            f.write(self.current_git_hash)
