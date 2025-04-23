import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    PreTrainedModel,
    TrainingArguments,
)
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from colpali_engine.collators import CorpusQueryCollator, VisualRetrieverCollator
from colpali_engine.loss.late_interaction_losses import (
    ColbertLoss,
)

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
        
        def init_distributed():
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",    # set MASTER_ADDR, MASTER_PORT externally
                )
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            return device, local_rank
        
        if dist.is_available() and dist.is_initialized():
            print("Distributed training is enabled.")
            device, local_rank = init_distributed()
        else:
            print("Distributed training is not enabled.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            local_rank = 0
        
        self.model.to(device)
        self.local_rank = local_rank
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[local_rank], output_device=local_rank
        )

    def train(self) -> None:
        if isinstance(self.collator, CorpusQueryCollator) and self.collator.mined_negatives:
            print("Training with hard negatives")
        else:
            print("Training with in-batch negatives")

        
        def gather_embeddings(tensor: torch.Tensor) -> torch.Tensor:
            world_size = dist.get_world_size()
            tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(tensor_list, tensor)
            return torch.cat(tensor_list, dim=0)
        

        sampler = DistributedSampler(self.dataset["train"])
        train_loader = DataLoader(
            self.dataset["train"],
            batch_size=self.config.tr_args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.collator,  # from colpali_engine.collators
        )

        if self.config.eval_dataset_loader is not None:
            eval_loader = DataLoader(
                self.dataset["validation"],
                batch_size=self.config.tr_args.per_device_eval_batch_size,
                collate_fn=self.collator,
            )
        else:   
            eval_loader = None
            print("No eval dataset provided. Skipping evaluation.")

        # opimizer adam
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.tr_args.learning_rate,
            weight_decay=self.config.tr_args.weight_decay,
        )
        # scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.tr_args.warmup_steps,
            gamma=0.1,
        )
        loss_fn = self.config.loss_func

        ## start training
        print("Starting training")
        for epoch in range(self.config.tr_args.num_train_epochs):
            print(f"Epoch {epoch + 1}/{self.config.tr_args.num_train_epochs}")
            sampler.set_epoch(epoch)
            self.model.train()
            for step, batch in enumerate(train_loader):
                # Move batch to device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                # Forward & get embeddings
                q_embed = self.model(**{"input_ids": batch["query_input_ids"],
                                "attention_mask": batch["query_attention_mask"]}).embeddings
                d_embed = self.model(**{k[4:]: v for k, v in batch.items() if k.startswith("doc_")}).embeddings
                # Optionally negative
                neg_embed = None
                if "neg_doc_input_ids" in batch:
                    neg_embed = self.model(**{k[8:]: v for k, v in batch.items() if k.startswith("neg_doc")}).embeddings
                # Gather
                q_global = gather_embeddings(q_embed)
                d_global = gather_embeddings(d_embed)
                n_global = gather_embeddings(neg_embed) if neg_embed is not None else None
                # Compute loss & backward
                loss = loss_fn(q_global, d_global, n_global)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
        

    def eval(self) -> None:
        raise NotImplementedError("Evaluation is not implemented yet.")

    def save(self):
        """
        Save the model with its training config, as well as the tokenizer and processor if provided.
        """
        self.model.save_pretrained(self.config.output_dir)
        self.config.processor.save_pretrained(self.config.output_dir)

        # Save git hash of the commit at beginning of training
        with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
            f.write(self.current_git_hash)
