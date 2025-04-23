from typing import Any, Dict, List, Optional, Union

from datasets import Dataset as HFDataset
from PIL.Image import Image
from torch.utils.data import Dataset


class ExternalDocumentCorpus:
    """
    Corpus class for handling retrieving with simple mapping.

    Args:
        corpus_data (List[Dict[str, Any]]): List of dictionaries containing doc data.
        doc_column (str): The key in the dictionary that contains the doc data.
        id_column (Optional[str]): Doc ID's entry key. Defaults to None for corpus index as the ID.
    """

    def __init__(self, corpus_data: HFDataset, docid_to_idx_mapping: Optional[Dict[str, int]] = None):
        """
        Initialize the corpus with the provided data.
        """
        self.corpus_data = corpus_data
        self.docid_to_idx_mapping = docid_to_idx_mapping

    def __len__(self) -> int:
        """
        Return the number of docs in the corpus.

        Returns:
            int: The number of docs in the corpus.
        """
        return len(self.corpus_data)

    def retrieve(self, docid: Any) -> Dict[str, Any]:
        """
        Get the corpus row from the given Doc ID.

        Args:
            docid (str): The id of the document.

        Returns:
            Dict[str, Any]: The row corresponding to the Doc ID.
        """
        if self.docid_to_idx_mapping is not None:
            return self.corpus_data[self.docid_to_idx.get(docid)]
        return self.corpus_data[docid]


class ColpaliEngineDataset(Dataset):
    # Output keys
    QUERY_KEY = "query"
    POS_TARGET_KEY = "pos_target"
    NEG_TARGET_KEY = "neg_target"

    def __init__(
        self,
        data: HFDataset,
        external_document_corpus: Optional[ExternalDocumentCorpus] = None,
    ):
        """
        Override the dataset class to handle your datasets.

        Args:
            data (Dict[str, List[Any]]): A dictionary containing the dataset samples.
            external_document_corpus (Optional[HFDataset]): An optional external document corpus to retrieve documents (images) from.
        """
        self.data = data
        self.external_document_corpus = external_document_corpus

        assert isinstance(
            self.data,
            (HFDataset, Dataset),
        ), "Data must be a Hugging Face Dataset or PyTorch Dataset"
        assert (
            isinstance(
                self.external_document_corpus,
                (HFDataset, Dataset),
            )
            or self.external_document_corpus is None
        ), "Corpus must be a Hugging Face Dataset or PyTorch Dataset"

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, Any]: The sample at the specified index.
        """
        sample = self.data[idx]

        if self.external_document_corpus is not None:
            return {
                self.QUERY_KEY: sample["query"],
                self.POS_TARGET_KEY: self.get_external_documents_from_docid(sample[self.pos_target_column]),
                self.NEG_TARGET_KEY: self.get_external_documents_from_docid(sample[self.neg_target_column])
                if self.neg_target_column
                else None,
            }
        return {
            self.QUERY_KEY: sample["query"],
            self.POS_TARGET_KEY: sample["pos_target"],
            self.NEG_TARGET_KEY: sample["neg_target"] if "neg_target" in sample else None,
        }

    def get_external_documents_from_docid(self, doc_ids) -> List[Union[str, Image]]:
        """
        Get the documents from the external corpus using the document ID.
        Args:
            doc_id (str): The document ID to retrieve.
        Returns:
            List[Union[str, Image]]: The documents retrieved from the external corpus.
        """

        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]

        return [self.external_document_corpus.retrieve(doc_id) for doc_id in doc_ids]
