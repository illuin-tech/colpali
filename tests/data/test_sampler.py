import pytest
import torch
from torch.utils.data import Dataset

from colpali_engine.data.sampler import SingleDatasetBatchSampler


class DummyDataset(Dataset):
    """
    Minimal PyTorch dataset that also supports `.take()`.
    The values it returns are irrelevant to the sampler; we only care about length.
    """

    def __init__(self, size: int, start: int = 0):
        self._data = list(range(start, start + size))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    # Simulate Arrow / HF dataset API used by the sampler
    def take(self, total_samples: int):
        # Keep the same starting offset so global indices stay monotonic
        return DummyDataset(total_samples, start=self._data[0])


# --------------------------------------------------------------------------- #
#                                Test helpers                                 #
# --------------------------------------------------------------------------- #
def dataset_boundaries(sampler):
    """Return a list of (lo, hi) index ranges, one per dataset, in global space."""
    cs = sampler.cumsum_sizes  # cumsum has an extra leading 0
    return [(cs[i], cs[i + 1]) for i in range(len(cs) - 1)]


def which_dataset(idx, boundaries):
    """Given a global idx, tell which dataset it belongs to (0‑based)."""
    for d, (lo, hi) in enumerate(boundaries):
        if lo <= idx < hi:
            return d
    raise ValueError(f"idx {idx} out of bounds")


# --------------------------------------------------------------------------- #
#                                   Tests                                     #
# --------------------------------------------------------------------------- #
def test_basic_iteration_and_len():
    """
    Two datasets, lengths 10 and 6, global batch size 4.

    Both datasets should be truncated (10→8, 6→4).  Expect 3 batches.
    """
    ds = [DummyDataset(10), DummyDataset(6)]
    gen = torch.Generator().manual_seed(123)
    sampler = SingleDatasetBatchSampler(ds, global_batch_size=4, generator=gen)

    batches = list(iter(sampler))

    # 1) __len__ matches actual number of batches
    assert len(batches) == len(sampler) == 3

    # 2) All samples are unique and count equals truncated total
    flat = [i for b in batches for i in b]
    assert len(flat) == len(set(flat)) == 12  # 8 + 4

    # 3) Every batch is exactly global_batch_size long
    assert all(len(b) == 4 for b in batches)


def test_single_dataset_per_batch():
    """
    Ensure that every yielded batch contains indices drawn from
    *one—and only one—dataset*.
    """
    ds = [DummyDataset(8), DummyDataset(8), DummyDataset(16)]
    sampler = SingleDatasetBatchSampler(ds, global_batch_size=4, generator=torch.Generator())

    boundaries = dataset_boundaries(sampler)

    for batch in sampler:
        d0 = which_dataset(batch[0], boundaries)
        # All indices in the batch must map to the same dataset ID
        assert all(which_dataset(i, boundaries) == d0 for i in batch)


def test_epoch_based_reshuffle_changes_order():
    """
    Calling set_epoch should reshuffle the internal order so that
    consecutive epochs produce different batch orderings.
    """
    ds = [DummyDataset(8), DummyDataset(8)]
    gen = torch.Generator().manual_seed(999)
    sampler = SingleDatasetBatchSampler(ds, global_batch_size=4, generator=gen)

    first_epoch = list(iter(sampler))

    sampler.set_epoch(1)
    second_epoch = list(iter(sampler))

    # Pure order comparison; contents are the same but order should differ
    assert first_epoch != second_epoch

    # Same epoch again → deterministic repeat
    sampler.set_epoch(1)
    repeat_epoch = list(iter(sampler))
    assert second_epoch == repeat_epoch


@pytest.mark.parametrize(
    "lengths,batch_size,expected_batches",
    [
        ([12], 4, 3),  # single dataset, perfect fit
        ([13], 4, 3),  # single dataset, truncated down
        ([7, 9], 4, 3),  # truncates both
        ([4, 4, 4], 4, 3),  # multiple, exact fit
    ],
)
def test_len_property_various_lengths(lengths, batch_size, expected_batches):
    datasets = [DummyDataset(n) for n in lengths]
    sampler = SingleDatasetBatchSampler(datasets, global_batch_size=batch_size)
    assert len(sampler) == expected_batches
