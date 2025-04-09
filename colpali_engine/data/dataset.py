from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from PIL.Image import Image
from torch.utils.data import Dataset

from colpali_engine.data.corpus import BaseCorpus


@dataclass
class IRColumn:
    column_name: str
    desc_column: List[str] = None
    corpus_column: Optional[str] = None
    corpus_metadata_columns: Optional[List[str]] = field(default_factory=list)


@dataclass
class Document:
    item: Union[str, Image]
    desc: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata to the document."""
        if self.metadata is None:
            self.metadata = {}
        self.metadata.update(metadata)


class IRDataset(Dataset):
    # Output keys
    QUERY_KEY = "query"
    POS_TARGET_KEY = "pos_target"
    NEG_TARGET_KEY = "neg_target"

    def __init__(
        self,
        data: Dict[str, List[Any]],
        corpus: Optional[BaseCorpus] = None,
        query_column: Optional[Union[str, IRColumn]] = "query",
        pos_target_column: Optional[Union[str, IRColumn]] = "image",
        neg_target_column: Optional[Union[str, IRColumn]] = None,
    ):
        """
        Initialize the dataset with a list of data samples.
        Each sample is a dictionary containing query and targets, whether in stored directly or in the corpus.

        Args:
            data (Dict[str, List[Any]]): A dictionary containing the dataset samples.
            corpus_data (Optional[BaseCorpus]): The corpus to retrieve documents from.
            query_column (Optional[Union[str, IRColumn]]): The column name for the query text.
            pos_target_column (Optional[Union[str, IRColumn]]): The column name for the positive images.
            neg_target_column (Optional[Union[str, IRColumn]]): The column name for the negative images.
        """
        self.data = data

        self.query_column = query_column
        self.pos_target_column = pos_target_column
        self.neg_target_column = neg_target_column

        if isinstance(query_column, str):
            self.query_column = IRColumn(query_column)
        if isinstance(pos_target_column, str):
            self.pos_target_column = IRColumn(pos_target_column)
        if isinstance(neg_target_column, str):
            self.neg_target_column = IRColumn(neg_target_column)

        self.corpus = corpus

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
        sample_data = self.data[idx]

        query = self.get_documents(sample_data, self.query_column)
        pos_target = self.get_documents(sample_data, self.pos_target_column)
        neg_target = self.get_documents(sample_data, self.neg_target_column) if self.neg_target_column else None

        return {
            self.QUERY_KEY: query,
            self.POS_TARGET_KEY: pos_target,
            self.NEG_TARGET_KEY: neg_target,
        }

    def get_documents(self, sample_data: Dict[str, Any], column: IRColumn) -> List[Document]:
        """
        Get the documents from the corpus or local path.

        Args:
            sample_data (Dict[str, Any]): The sample data.
            column (IRColumn): The column to retrieve documents from.
        Returns:
            List[Document]: A list of documents.
        """
        docs_or_ids = sample_data[column.column_name]

        if not isinstance(docs_or_ids, list):
            docs_or_ids = [docs_or_ids]

        return [self._collate_doc(doc_or_id, column=column) for doc_or_id in docs_or_ids]

    def _collate_doc(self, doc_or_id: Union[str, Union[str, Image]], column: IRColumn) -> Document:
        """
        Collate a document or ID into a Document object if stored in the corpus, else return the dataset entry.
        """
        if isinstance(doc_or_id, Image):
            return doc_or_id

        retrieve_doc = self.corpus is not None and column.corpus_column is not None

        if retrieve_doc:
            # If the corpus is provided, retrieve the document using the corpus
            corpus_data = self.corpus.retrieve(doc_or_id)
            return Document(
                item=corpus_data[column.corpus_column],
                metadata={col: corpus_data[col] for col in corpus_data.keys() if col in column.corpus_metadata_columns},
            )
        else:
            return doc_or_id
