import torch
from datasets import DatasetDict
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import Trainer, is_datasets_available
from transformers.trainer_utils import seed_worker

from colpali_engine.data.sampler import SingleDatasetBatchSampler

from torch.distributed.nn.functional import all_gather  # PyTorch ≥ 2.1


def concat_all_gather(t: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.cat(all_gather(t), dim=0)  # keeps grad graph
    return t


class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, *args, **kwargs):
        if isinstance(kwargs["train_dataset"], DatasetDict):
            dataset_list = list(kwargs["train_dataset"].values())
        elif isinstance(kwargs["train_dataset"], list):
            dataset_list = kwargs["train_dataset"]
        else:
            dataset_list = None

        if dataset_list is not None:
            kwargs["train_dataset"] = ConcatDataset(dataset_list)

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
            offset = 0
            if self.accelerator.num_processes > 1 and self.accelerator.sync_gradients:
                # gather docs across all processes
                doc_outputs = concat_all_gather(doc_outputs)
                rank = self.accelerator.process_index
                offset = rank * num_items_in_batch if num_items_in_batch is not None else 0
                
            loss = self.loss_func(query_outputs, doc_outputs, offset=offset)
            
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
