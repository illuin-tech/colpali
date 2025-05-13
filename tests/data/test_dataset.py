import pytest
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
#  Adjust the next line to the real module path of Corpus & ColPaliEngineDataset
from colpali_engine.data import ColPaliEngineDataset, Corpus

# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#                            Helper test fixtures                              #
# --------------------------------------------------------------------------- #
class DummyMapDataset(Dataset):
    """
    Minimal map‑style dataset with a `.take()` method, so we can test
    ColPaliEngineDataset.take() without the HuggingFace `datasets` package.
    """

    def __init__(self, samples):
        self._samples = list(samples)

    # Map‑style API
    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]

    # Arrow‑style `.take` API (returns the **same** type)
    def take(self, n):
        return DummyMapDataset(self._samples[:n])


# Common sample structures -------------------------------------------------- #
@pytest.fixture
def basic_corpus_data():
    return [{"doc": f"document_{i}"} for i in range(3)]  # indices 0,1,2


@pytest.fixture
def id_mapping():
    return {"A": 0, "B": 1, "C": 2}


@pytest.fixture
def basic_corpus(basic_corpus_data):
    return Corpus(corpus_data=basic_corpus_data)


@pytest.fixture
def mapped_corpus(basic_corpus_data, id_mapping):
    return Corpus(corpus_data=basic_corpus_data, docid_to_idx_mapping=id_mapping)


# Data for ColPaliEngineDataset --------------------------------------------- #
@pytest.fixture
def raw_dataset_data():
    """
    Returns a list of 2 samples:

    idx 0 → pos_target is a *scalar* id
    idx 1 → pos_target and neg_target are lists
    """
    return [
        {
            "query": "what is doc zero?",
            "pos_target": 0,
            # intentionally omit neg_target
        },
        {
            "query": "compare docs",
            "pos_target": [1, 2],
            "neg_target": 0,
        },
    ]


# --------------------------------------------------------------------------- #
#                                 Corpus tests                                #
# --------------------------------------------------------------------------- #
def test_corpus_len_and_index_retrieve(basic_corpus):
    assert len(basic_corpus) == 3
    assert basic_corpus.retrieve(2) == "document_2"


def test_corpus_retrieve_with_id_mapping(mapped_corpus):
    assert mapped_corpus.retrieve("B") == "document_1"
    assert mapped_corpus.retrieve("A") == "document_0"


# --------------------------------------------------------------------------- #
#                      ColPaliEngineDataset – without corpus                  #
# --------------------------------------------------------------------------- #
def test_dataset_item_scalar_pos_no_neg(raw_dataset_data):
    ds = ColPaliEngineDataset(raw_dataset_data)

    sample = ds[0]
    assert sample[ColPaliEngineDataset.QUERY_KEY] == "what is doc zero?"
    # scalar pos_target should be converted to list
    assert sample[ColPaliEngineDataset.POS_TARGET_KEY] == [0]
    # neg_target absent → None
    assert sample[ColPaliEngineDataset.NEG_TARGET_KEY] is None


def test_dataset_item_list_pos_and_neg(raw_dataset_data):
    ds = ColPaliEngineDataset(raw_dataset_data)

    sample = ds[1]
    assert sample[ColPaliEngineDataset.POS_TARGET_KEY] == [1, 2]
    # string/int neg_target should be wrapped in list
    assert sample[ColPaliEngineDataset.NEG_TARGET_KEY] == [0]


# --------------------------------------------------------------------------- #
#                     ColPaliEngineDataset – with corpus                      #
# --------------------------------------------------------------------------- #
def test_dataset_resolves_targets_via_corpus(raw_dataset_data, basic_corpus):
    # Here raw_dataset_data uses *indices* 0,1,2 that match the corpus rows
    ds = ColPaliEngineDataset(raw_dataset_data, corpus=basic_corpus)

    sample = ds[1]
    # Expect actual documents, not IDs
    assert sample[ColPaliEngineDataset.POS_TARGET_KEY] == ["document_1", "document_2"]
    assert sample[ColPaliEngineDataset.NEG_TARGET_KEY] == ["document_0"]


def test_dataset_resolves_targets_with_id_mapping(raw_dataset_data, mapped_corpus):
    """
    Map pos/neg IDs 'A','B','C' through the corpus mapping.
    """
    data_with_ids = [
        {
            "query": "give me A",
            "pos_target": "A",
        },
        {
            "query": "give me all",
            "pos_target": ["B"],
            "neg_target": ["C"],
        },
    ]
    ds = ColPaliEngineDataset(data_with_ids, corpus=mapped_corpus)

    s0 = ds[0]
    assert s0[ColPaliEngineDataset.POS_TARGET_KEY] == ["document_0"]

    s1 = ds[1]
    assert s1[ColPaliEngineDataset.POS_TARGET_KEY] == ["document_1"]
    assert s1[ColPaliEngineDataset.NEG_TARGET_KEY] == ["document_2"]


# --------------------------------------------------------------------------- #
#                              .take() behaviour                              #
# --------------------------------------------------------------------------- #
def test_take_returns_subset(raw_dataset_data, basic_corpus):
    # Use DummyMapDataset so that take() is available
    wrapped = DummyMapDataset(raw_dataset_data)
    ds = ColPaliEngineDataset(wrapped, corpus=basic_corpus)

    sub_ds = ds.take(1)

    # Returned object should be the *same* class
    assert isinstance(sub_ds, ColPaliEngineDataset)
    assert len(sub_ds) == 1

    # First (and only) sample should match original idx 0
    assert sub_ds[0][ColPaliEngineDataset.QUERY_KEY] == "what is doc zero?"
