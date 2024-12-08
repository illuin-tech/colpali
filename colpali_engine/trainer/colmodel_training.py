import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from datasets import concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)

from colpali_engine.collators import CorpusQueryCollator, VisualRetrieverCollator
from colpali_engine.loss.late_interaction_losses import (
    ColbertCELoss,
)
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.trainer.eval_utils import CustomRetrievalEvaluator
from colpali_engine.utils.gpu_stats import print_gpu_utilization, print_summary
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


@dataclass
class ColModelTrainingConfig:
    model: PreTrainedModel
    tr_args: TrainingArguments = None
    output_dir: str = None
    max_length: int = 256
    run_eval: bool = True
    run_train: bool = True
    peft_config: Optional[LoraConfig] = None
    processor: BaseVisualRetrieverProcessor = None
    tokenizer: PreTrainedTokenizer = None
    loss_func: Optional[Callable] = ColbertCELoss()
    dataset_loading_func: Optional[Callable] = None
    eval_dataset_loader: Optional[Dict[str, Callable]] = None
    pretrained_peft_model_name_or_path: Optional[str] = None

    def __post_init__(self):
        """
        Initialize the model and tokenizer if not provided
        """
        if self.output_dir is None:
            sanitized_name = str(self.model.name_or_path).replace("/", "_")
            self.output_dir = f"./models/{sanitized_name}"

        if self.tr_args is None:
            self.tr_args = TrainingArguments(output_dir=self.output_dir)
        elif self.tr_args.output_dir is None:
            self.tr_args.output_dir = self.output_dir

        # cast if string
        if isinstance(self.tr_args.learning_rate, str):
            self.tr_args.learning_rate = float(self.tr_args.learning_rate)
        self.tr_args.remove_unused_columns = False

        if self.processor is None and self.tokenizer is None:
            print("Using textual model tokenization")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.name_or_path)

        if self.pretrained_peft_model_name_or_path is not None:
            self.model.load_adapter(self.pretrained_peft_model_name_or_path)

            print(f"Loaded pretrained adapter from {self.pretrained_peft_model_name_or_path}")

        if self.peft_config is not None:
            print("Configurating PEFT model")
            if self.processor is None:
                # Might be deprecated - use the "else" branch
                self.model = prepare_model_for_kbit_training(self.model)  # use_gradient_checkpointing=True
                # self.model.enable_input_require_grads()
                self.model = get_peft_model(self.model, self.peft_config)
                self.model.print_trainable_parameters()
            else:
                if self.pretrained_peft_model_name_or_path is None:
                    # self.model.add_adapter(self.peft_config)
                    # self.model.enable_adapters()
                    self.model = get_peft_model(self.model, self.peft_config)
                    self.model.print_trainable_parameters()
                else:
                    print(f"Adapter already loaded from {self.pretrained_peft_model_name_or_path}. Not overwriting.")

    print_gpu_utilization()


class ColModelTraining:
    def __init__(self, config: ColModelTrainingConfig) -> None:
        self.config = config
        self.model = self.config.model
        self.dataset = self.config.dataset_loading_func()
        if isinstance(self.dataset, Tuple):
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
            self.collator = VisualRetrieverCollator(
                processor=self.config.processor,
                max_length=self.config.max_length,
            )
        self.current_git_hash = os.popen("git rev-parse HEAD").read().strip()
        self.retrieval_evaluator = CustomRetrievalEvaluator()

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

        result = trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint)
        print_summary(result)

    def eval_dataset(self, test_dataset):
        self.model.eval()

        idx_with_query = [idx for idx, sample in enumerate(test_dataset["query"]) if sample is not None]
        idx_without_query = [idx for idx, sample in enumerate(test_dataset["query"]) if sample is None]

        dataloader_with_query = DataLoader(
            test_dataset.select(idx_with_query),
            batch_size=self.config.tr_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )
        dataloader_without_query = DataLoader(
            test_dataset.select(idx_without_query),
            batch_size=self.config.tr_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        # dataset is ordered so that non-null queries come first
        test_dataset = concatenate_datasets(
            [test_dataset.select(idx_with_query), test_dataset.select(idx_without_query)]
        )

        relevant_docs = {}
        docidx_2_docid = {}
        qsidx_2_query = []
        for idx, sample in enumerate(test_dataset):
            doc_id = sample["image_filename"] if "image_filename" in sample else str(hash(sample["doc"]))
            # query_id = sample["query_id"] if "query_id" in sample else str(hash(sample["query"]))
            if sample["query"] is not None:
                relevant_docs[str(idx)] = {doc_id: 1}
                qsidx_2_query.append(str(idx))
            docidx_2_docid[str(idx)] = doc_id

        qs = []
        ps = []

        device = self.model.device
        with torch.no_grad():
            for dataloader in [dataloader_with_query, dataloader_without_query]:
                for batch in tqdm(dataloader):
                    # feed only kwargs with 'doc_' prefix
                    output = self.model(**{k[4:]: v.to(device) for k, v in batch.items() if k.startswith("doc")})

                    from colpali_engine.models.paligemma.colpali_duo.modeling_colpali_duo import ColPaliDuoModelOutput

                    # hacky way to get the embeddings
                    if isinstance(output, ColPaliDuoModelOutput):
                        if True:
                            doc = output.single_vec_emb
                        else:
                            doc = output.multi_vec_emb

                    ps.extend(list(torch.unbind(doc.to("cpu"))))

                    if "query_input_ids" in batch:
                        query = self.model(
                            input_ids=batch["query_input_ids"].to(device),
                            attention_mask=batch["query_attention_mask"].to(device),
                        )

                        if isinstance(query, ColPaliDuoModelOutput):
                            if True:
                                query = query.single_vec_emb
                            else:
                                query = query.multi_vec_emb

                        # variable len
                        qs.extend(list(torch.unbind(query.to("cpu"))))

        print("Embeddings computed, evaluating")
        scores = self.config.processor.score(qs, ps, device=self.model.device)
        # scores is 2d array of shape (n_queries, n_docs)
        # turn it into a dict
        results = {}
        assert scores.shape[0] == len(qsidx_2_query)
        for idx, scores_per_query in enumerate(scores):
            results[qsidx_2_query[idx]] = {
                docidx_2_docid[str(docidx)]: float(score) for docidx, score in enumerate(scores_per_query)
            }

        # evaluate
        metrics = self.retrieval_evaluator.compute_mteb_metrics(relevant_docs, results)
        print("MTEB metrics:", metrics)

        return metrics

    def eval(self) -> None:
        all_metrics = {}
        try:
            print("Evaluating on validation set")
            metrics = self.eval_dataset(self.dataset["test"])
            print(f"Metrics for validation set: {metrics}")
            all_metrics["validation_set"] = metrics
        except Exception as e:
            print(f"Error evaluating validation set: {e}")

        # switching to normal collator
        self.collator = VisualRetrieverCollator(
            processor=self.config.processor,
            max_length=self.config.max_length,
        )
        if self.config.eval_dataset_loader is not None:
            for test_name, test_dataset_loading_func in self.config.eval_dataset_loader.items():
                print(f"Evaluating {test_name}")
                test_ds = test_dataset_loading_func()
                metrics = self.eval_dataset(test_ds)
                all_metrics[test_name] = metrics
                print(f"Metrics for {test_name}: {metrics}")

                # checkpoint dumps
                with open(f"{self.config.output_dir}/results.json", "w") as f:
                    json.dump(all_metrics, f)

        # save results as json
        with open(f"{self.config.output_dir}/results.json", "w") as f:
            json.dump(all_metrics, f)

    def save(self, config_file):
        # save model
        self.model.save_pretrained(self.config.output_dir)
        if self.config.tokenizer is not None:
            self.config.tokenizer.save_pretrained(self.config.output_dir)
        if self.config.processor is not None:
            self.config.processor.save_pretrained(self.config.output_dir)  # save config

        # copy-paste the yml file with os
        os.system(f"cp {config_file} {self.config.output_dir}/training_config.yml")

        # save git hash of the commit at beginning of training
        with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
            f.write(self.current_git_hash)
