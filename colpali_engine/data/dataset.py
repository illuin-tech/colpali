import random
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import Dataset

Document = Union[str, Image.Image]


class Corpus:
    """
    Corpus class for handling retrieving with simple mapping.
    This class is meant to be overridden by the user to handle their own corpus.

    Args:
        corpus_data (List[Dict[str, Any]]): List of dictionaries containing doc data.
        docid_to_idx_mapping (Optional[Dict[str, int]]): Optional mapping from doc IDs to indices.
    """

    def __init__(
        self,
        corpus_data: List[Dict[str, Any]],
        docid_to_idx_mapping: Optional[Dict[str, int]] = None,
        doc_column_name: str = "doc",
    ):
        """
        Initialize the corpus with the provided data.
        """
        self.corpus_data = corpus_data
        self.docid_to_idx_mapping = docid_to_idx_mapping
        self.doc_column_name = doc_column_name

        assert isinstance(
            self.corpus_data,
            (list, Dataset, HFDataset),
        ), "Corpus data must be a map-style dataset"

        assert self.doc_column_name in self.corpus_data[0], f"Corpus data must contain a column {self.doc_column_name}."

    def __len__(self) -> int:
        """
        Return the number of docs in the corpus.

        Returns:
            int: The number of docs in the corpus.
        """
        return len(self.corpus_data)

    def retrieve(self, docid: Any) -> Document:
        """
        Get the corpus row from the given Doc ID.

        Args:
            docid (str): The id of the document.

        Returns:
            Document: The document retrieved from the corpus.
        """
        if self.docid_to_idx_mapping is not None:
            doc_idx = self.docid_to_idx_mapping[docid]
        else:
            doc_idx = docid
        return self.corpus_data[doc_idx][self.doc_column_name]


class ColPaliEngineDataset(Dataset):
    # Output keys
    QUERY_KEY = "query"
    POS_TARGET_KEY = "pos_target"
    NEG_TARGET_KEY = "neg_target"

    def __init__(
        self,
        data: List[Dict[str, Any]],
        corpus: Optional[Corpus] = None,
        query_column_name: str = "query",
        pos_target_column_name: str = "pos_target",
        neg_target_column_name: str = None,
        num_negatives: int = 3,
    ):
        """
        Initialize the dataset with the provided data and external document corpus.

        Args:
            data (Dict[str, List[Any]]): A dictionary containing the dataset samples.
            corpus (Optional[Corpus]): An optional external document corpus to retrieve
            documents (images) from.
        """
        self.data = data
        self.corpus = corpus

        # Column args
        self.query_column_name = query_column_name
        self.pos_target_column_name = pos_target_column_name
        self.neg_target_column_name = neg_target_column_name

        self.num_negatives = num_negatives
        assert isinstance(
            self.data,
            (list, Dataset, HFDataset),
        ), "Data must be a map-style dataset"

        assert self.query_column_name in self.data[0], f"Data must contain the {self.query_column_name} column"
        assert self.pos_target_column_name in self.data[0], f"Data must contain a {self.pos_target_column_name} column"
        if self.neg_target_column_name is not None:
            assert self.neg_target_column_name in self.data[0], (
                f"Data must contain a {self.neg_target_column_name} column"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]

        query = sample[self.query_column_name]

        pos_targets = sample[self.pos_target_column_name]
        if not isinstance(pos_targets, list):
            pos_targets = [pos_targets]

        if self.neg_target_column_name is not None:
            neg_targets = sample[self.neg_target_column_name]
            if not isinstance(neg_targets, list):
                neg_targets = [neg_targets]
        else:
            neg_targets = None

        # If an external document corpus is provided, retrieve the documents from it.
        if self.corpus is not None:
            pos_targets = [self.corpus.retrieve(doc_id) for doc_id in pos_targets]
            if neg_targets is not None:
                # to avoid oveflowing CPU memory
                if len(neg_targets) > self.num_negatives:
                    neg_targets = random.sample(neg_targets, self.num_negatives)
                neg_targets = [self.corpus.retrieve(doc_id) for doc_id in neg_targets]

        return {
            self.QUERY_KEY: query,
            self.POS_TARGET_KEY: pos_targets,
            self.NEG_TARGET_KEY: neg_targets,
        }

    def take(self, n: int) -> "ColPaliEngineDataset":
        """
        Take the first n samples from the dataset.

        Args:
            n (int): The number of samples to take.

        Returns:
            ColPaliEngineDataset: A new dataset containing the first n samples.
        """
        return self.__class__(
            self.data.take(n),
            self.corpus,
            self.query_column_name,
            self.pos_target_column_name,
            self.neg_target_column_name,
        )
