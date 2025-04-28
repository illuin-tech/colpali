from typing import Any, Dict, List, Optional, Union

from datasets import Dataset as HFDataset
from PIL.Image import Image
from torch.utils.data import Dataset

Document = Union[str, Image]


class ExternalDocumentCorpus:
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

        assert "doc" in self.corpus_data[0], (
            f"Corpus data must contain a 'doc' column. Got: {self.corpus_data[0].keys()}"
        )

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
            return self.corpus_data[self.docid_to_idx_mapping.get(docid)][self.doc_column_name]
        return self.corpus_data[docid][self.doc_column_name]


class ColPaliEngineDataset(Dataset):
    # Output keys
    QUERY_KEY = "query"
    POS_TARGET_KEY = "pos_target"
    NEG_TARGET_KEY = "neg_target"

    def __init__(
        self,
        data: List[Dict[str, Any]],
        external_document_corpus: Optional[ExternalDocumentCorpus] = None,
    ):
        """
        Initialize the dataset with the provided data and external document corpus.

        Args:
            data (Dict[str, List[Any]]): A dictionary containing the dataset samples.
            external_document_corpus (Optional[HFDataset]): An optional external document corpus to retrieve
            documents (images) from.
        """
        self.data = data
        self.external_document_corpus = external_document_corpus

        assert isinstance(
            self.data,
            (list, Dataset, HFDataset),
        ), "Data must be a map-style dataset"

        assert self.QUERY_KEY in self.data[0], f"Data must contain a {self.QUERY_KEY} column"
        assert self.POS_TARGET_KEY in self.data[0], f"Data must contain a {self.POS_TARGET_KEY} column"

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]

        if self.external_document_corpus is not None:
            return {
                self.QUERY_KEY: sample[self.QUERY_KEY],
                self.POS_TARGET_KEY: self.get_external_documents_from_docid(sample[self.POS_TARGET_KEY]),
                self.NEG_TARGET_KEY: (
                    self.get_external_documents_from_docid(sample[self.NEG_TARGET_KEY])
                    if self.NEG_TARGET_KEY in sample
                    else None
                ),
            }

        return {
            self.QUERY_KEY: sample[self.QUERY_KEY],
            self.POS_TARGET_KEY: sample[self.POS_TARGET_KEY],
            self.NEG_TARGET_KEY: sample[self.NEG_TARGET_KEY] if self.NEG_TARGET_KEY in sample else None,
        }

    def get_external_documents_from_docid(self, doc_ids) -> List[Document]:
        """
        Get the documents from the external corpus using the document ID.
        Args:
            doc_id (str): The document ID to retrieve.
        Returns:
            List[Document]: The documents retrieved from the external corpus.
        """

        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]

        return [self.external_document_corpus.retrieve(doc_id) for doc_id in doc_ids]
