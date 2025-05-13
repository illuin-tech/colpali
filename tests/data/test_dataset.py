import pytest
from torch.utils.data import Dataset

from colpali_engine.data import ColPaliEngineDataset, Corpus


# --------------------------------------------------------------------------- #
#                              Helper utilities                               #
# --------------------------------------------------------------------------- #
class DummyMapDataset(Dataset):
    """
    Minimal map‑style dataset that includes a `.take()` method so we can
    exercise ColPaliEngineDataset.take() without depending on HF datasets.
    """

    def __init__(self, samples):
        self._samples = list(samples)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]

    def take(self, n):
        return DummyMapDataset(self._samples[:n])


# --------------------------------------------------------------------------- #
#                             Fixtures & samples                              #
# --------------------------------------------------------------------------- #
@pytest.fixture
def corpus():
    data = [{"doc": f"doc_{i}"} for i in range(3)]
    return Corpus(corpus_data=data)


@pytest.fixture
def data_no_neg():
    """3 samples – *no* neg_target column at all."""
    return [
        {"query": "q0", "pos_target": 0},
        {"query": "q1", "pos_target": [1]},
        {"query": "q2", "pos_target": [2]},
    ]


@pytest.fixture
def data_with_neg():
    """2 samples – every sample has a neg_target column."""
    return [
        {"query": "q0", "pos_target": 1, "neg_target": 0},
        {"query": "q1", "pos_target": [2], "neg_target": [0, 1]},
    ]


# --------------------------------------------------------------------------- #
#                          Tests – NO negatives case                          #
# --------------------------------------------------------------------------- #
def test_no_negatives_basic(data_no_neg):
    ds = ColPaliEngineDataset(data_no_neg)  # neg_target_column_name defaults to None
    assert len(ds) == 3

    sample = ds[0]
    assert sample[ColPaliEngineDataset.QUERY_KEY] == "q0"
    assert sample[ColPaliEngineDataset.POS_TARGET_KEY] == [0]
    # NEG_TARGET_KEY should be None
    assert sample[ColPaliEngineDataset.NEG_TARGET_KEY] is None


def test_no_negatives_with_corpus_resolution(data_no_neg, corpus):
    ds = ColPaliEngineDataset(data_no_neg, corpus=corpus)
    s1 = ds[1]
    # pos_target indices 1 should be resolved to the actual doc string
    assert s1[ColPaliEngineDataset.POS_TARGET_KEY] == ["doc_1"]
    # still no negatives
    assert s1[ColPaliEngineDataset.NEG_TARGET_KEY] is None


# --------------------------------------------------------------------------- #
#                           Tests – WITH negatives case                       #
# --------------------------------------------------------------------------- #
def test_with_negatives_basic(data_with_neg):
    ds = ColPaliEngineDataset(
        data_with_neg,
        neg_target_column_name="neg_target",
    )
    assert len(ds) == 2

    s0 = ds[0]
    assert s0[ColPaliEngineDataset.POS_TARGET_KEY] == [1]
    assert s0[ColPaliEngineDataset.NEG_TARGET_KEY] == [0]


def test_with_negatives_and_corpus(data_with_neg, corpus):
    ds = ColPaliEngineDataset(
        data_with_neg,
        corpus=corpus,
        neg_target_column_name="neg_target",
    )
    s1 = ds[1]
    # pos 2 -> "doc_2", negs 0,1 -> "doc_0", "doc_1"
    assert s1[ColPaliEngineDataset.POS_TARGET_KEY] == ["doc_2"]
    assert s1[ColPaliEngineDataset.NEG_TARGET_KEY] == ["doc_0", "doc_1"]


# --------------------------------------------------------------------------- #
#                    Tests for mixed / inconsistent scenarios                 #
# --------------------------------------------------------------------------- #
def test_error_if_neg_column_specified_but_missing(data_no_neg):
    """All samples must include the column when neg_target_column_name is given."""
    with pytest.raises(AssertionError):
        ds = ColPaliEngineDataset(  # noqa: F841
            data_no_neg,
            neg_target_column_name="neg_target",
        )
        _ = ds[0]  # force __getitem__


def test_error_if_data_mix_neg_and_non_neg(data_with_neg, data_no_neg):
    """A mixed dataset (some samples without neg_target) should fail."""
    mixed = data_with_neg + data_no_neg
    # The first sample *does* have neg_target, so __init__ succeeds.
    ds = ColPaliEngineDataset(
        mixed,
        neg_target_column_name="neg_target",
    )
    # Accessing a sample lacking the column should raise.
    with pytest.raises(KeyError):
        _ = ds[len(data_with_neg)]  # first sample from the 'no_neg' part


# --------------------------------------------------------------------------- #
#                          .take() works in both modes                        #
# --------------------------------------------------------------------------- #
def test_take_returns_subset(data_no_neg):
    wrapped = DummyMapDataset(data_no_neg)
    ds = ColPaliEngineDataset(wrapped)

    sub_ds = ds.take(1)

    assert isinstance(sub_ds, ColPaliEngineDataset)
    assert len(sub_ds) == 1
    # Make sure we can still index
    _ = sub_ds[0]
