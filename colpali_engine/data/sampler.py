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
        # Reset state
        self.current_positions = [0] * len(self.datasets)
        self.available_datasets = list(range(len(self.datasets)))
        self.current_data_lengths = [size for size in self.dataset_sizes]  # full length, never shrinks

        while self.available_datasets:
            # Build probabilities for available datasets only
            lengths = [self.current_data_lengths[i] for i in self.available_datasets]
            total_length = sum(lengths)
            if total_length <= 0:
                break  # nothing left to sample

            probs = torch.tensor(lengths, dtype=torch.float) / total_length

            # Pick dataset
            dataset_idx_in_available = torch.multinomial(probs, num_samples=1, generator=self.generator).item()
            dataset_idx = self.available_datasets[dataset_idx_in_available]

            # Fetch batch
            dataset_indices = self.indices_per_dataset[dataset_idx]
            current_pos = self.current_positions[dataset_idx]
            end_pos = current_pos + self.global_batch_size

            if end_pos <= self.max_positions[dataset_idx]:
                batch_indices = [idx + self.cumsum_sizes[dataset_idx] for idx in dataset_indices[current_pos:end_pos]]
                self.current_positions[dataset_idx] = end_pos
                self.current_data_lengths[dataset_idx] = self.dataset_sizes[dataset_idx] - end_pos

                # Remove if exhausted
                if end_pos >= self.max_positions[dataset_idx]:
                    self.available_datasets.remove(dataset_idx)

                yield batch_indices
            else:
                # Not enough for a full batch
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
