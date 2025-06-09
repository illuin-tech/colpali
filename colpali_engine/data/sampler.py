from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset


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
        self.initial_seed = self.generator.initial_seed()

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
        torch_gen = torch.Generator()

        # Set seed based on epoch to ensure different shuffling each epoch
        new_seed = self.initial_seed + epoch
        torch_gen.manual_seed(new_seed)
        self.generator.manual_seed(new_seed)

        # Reshuffle indices for each dataset
        self.indices_per_dataset = [torch.randperm(size, generator=torch_gen).tolist() for size in self.dataset_sizes]

    @property
    def batch_size(self) -> int:
        return self.global_batch_size

    def __len__(self) -> int:
        return sum(size // self.global_batch_size for size in self.dataset_sizes)
