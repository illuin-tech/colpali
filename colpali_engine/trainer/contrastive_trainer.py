from typing import Iterator, List, Optional

import numpy as np
import torch
from datasets import DatasetDict
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, Dataset
from transformers import Trainer, is_datasets_available
from transformers.trainer_utils import seed_worker


class SingleDatasetBatchSampler(BatchSampler):
    """
    A batch sampler that samples from a single dataset per batch and handles distribution across GPUs.

    Args:
        datasets (List[Dataset]): List of datasets to sample from
        batch_size (int): Global batch size (will be divided across GPUs)
        drop_last (bool): Whether to drop the last incomplete batch
        generator (Optional[torch.Generator]): Random number generator
    """

    def __init__(
        self,
        datasets: List[Dataset],
        global_batch_size: int,
        drop_last: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        self.datasets = datasets
        self.global_batch_size = global_batch_size
        self.drop_last = drop_last
        self.generator = generator or torch.Generator()

        # Calculate dataset sizes and create index mappings
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        #### get start of each dataset #####
        self.cumsum_sizes = np.cumsum([0] + self.dataset_sizes).tolist()
        self.total_size = sum(self.dataset_sizes)

        # Create shuffled indices for each dataset
        self.indices_per_dataset = [
            torch.randperm(size, generator=self.generator).tolist() for size in self.dataset_sizes
        ]
        self.current_positions = [0] * len(datasets)

        self.available_datasets = list(range(len(datasets)))
        self.max_positions = [(size // self.global_batch_size) * self.global_batch_size for size in self.dataset_sizes]

    def __iter__(self) -> Iterator[List[int]]:
        # Reset current positions and available datasets
        self.current_positions = [0] * len(self.datasets)
        self.available_datasets = list(range(len(self.datasets)))

        while self.available_datasets:
            # Randomly select from available datasets
            dataset_idx_index = torch.randint(len(self.available_datasets), size=(1,), generator=self.generator).item()
            dataset_idx = self.available_datasets[dataset_idx_index]

            # Get indices for the current dataset
            dataset_indices = self.indices_per_dataset[dataset_idx]
            current_pos = self.current_positions[dataset_idx]

            # Check if we have enough samples for a full batch
            end_pos = current_pos + self.global_batch_size

            if end_pos <= self.max_positions[dataset_idx]:
                # Get batch indices
                batch_indices = [idx + self.cumsum_sizes[dataset_idx] for idx in dataset_indices[current_pos:end_pos]]

                # Update position
                self.current_positions[dataset_idx] = end_pos

                # If dataset is exhausted, remove from available datasets
                if end_pos >= self.max_positions[dataset_idx]:
                    self.available_datasets.remove(dataset_idx)

                yield batch_indices
            else:
                # This dataset doesn't have enough samples for another batch
                self.available_datasets.remove(dataset_idx)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler.

        Args:
            epoch (int): Epoch number
        """
        # Set seed based on epoch to ensure different shuffling each epoch
        torch_gen = torch.Generator()
        torch_gen.manual_seed(self.generator.initial_seed() + epoch)

        # Reshuffle indices for each dataset
        self.indices_per_dataset = [torch.randperm(size, generator=torch_gen).tolist() for size in self.dataset_sizes]

    @property
    def batch_size(self) -> int:
        return self.global_batch_size

    def __len__(self) -> int:
        return sum(size // self.global_batch_size for size in self.dataset_sizes)


class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, *args, **kwargs):
        if isinstance(kwargs["train_dataset"], DatasetDict):
            dataset_list = list(kwargs["train_dataset"].values())
            # TODO: This is quite hacky, we should find a better way to handle this
            # round down each dataset if not divible by global batch size
            batch_size = kwargs["args"].train_batch_size
            for i in range(len(dataset_list)):
                if len(dataset_list[i]) % batch_size != 0:
                    total_samples = (len(dataset_list[i]) // batch_size) * batch_size
                    dataset_list[i] = dataset_list[i].take(total_samples)

            kwargs["train_dataset"] = ConcatDataset(dataset_list)
        else:
            dataset_list = None

        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.is_vision_model = is_vision_model  # Unused argument, will be removed in 0.4.0
        self.args.remove_unused_columns = False  # Safety, don't remove dataset columns from dataloader
        self.dataset_list = dataset_list

    def get_train_dataloader(self):
        ######## adapted from Transformers Trainer (gross) ########
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.dataset_list is None:
            return super().get_train_dataloader()

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            ######### don't set batch size, mutually exclusive from batch sampler ######
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            ###### batch_sampler set instead of sampler in trainer code #######
            dataloader_params["batch_sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        return dataloader

    def _get_train_sampler(self):
        if self.dataset_list is None:
            return super()._get_train_sampler()

        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return SingleDatasetBatchSampler(
            self.dataset_list,
            self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            generator=generator,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
        # feed only kwargs with 'doc_' prefix
        doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
        if "neg_doc_input_ids" in inputs:
            neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
            loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
            return (loss, (query_outputs, doc_outputs, neg_doc_outputs)) if return_outputs else loss

        if "labels" in inputs:
            loss = self.loss_func(query_outputs, doc_outputs, inputs["labels"])
        else:
            loss = self.loss_func(query_outputs, doc_outputs)
        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")

        with torch.no_grad():
            # feed only kwargs with 'doc_' prefix
            doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
            query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
            if "neg_doc_input_ids" in inputs:
                neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
                loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
                return loss, None, None

            if "labels" in inputs:
                loss = self.loss_func(query_outputs, doc_outputs, inputs["labels"])
            else:
                loss = self.loss_func(query_outputs, doc_outputs)
            return loss, None, None
