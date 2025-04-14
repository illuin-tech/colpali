from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset
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

    @classmethod
    def from_hf(
        cls,
        dataset_name: str,
        corpus: BaseCorpus = None,
        query_column: Optional[Union[str, IRColumn]] = "query",
        pos_target_column: Optional[Union[str, IRColumn]] = "image",
        neg_target_column: Optional[Union[str, IRColumn]] = None,
        **kwargs,
    ) -> "IRDataset":
        """Create a dataset from a Hugging Face dataset.

        Args:
            dataset_name (str): The name of the Hugging Face dataset.
            corpus (BaseCorpus): The corpus to retrieve documents from.
            query_column (Optional[Union[str, IRColumn]]): The column name for the query text.
            pos_target_column (Optional[Union[str, IRColumn]]): The column name for the positive images.
            neg_target_column (Optional[Union[str, IRColumn]]): The column name for the negative images.
            **kwargs: Additional arguments to pass to the HF dataset loading function.

        Returns:
            IRDataset: An instance of the IRDataset class.
        """
        return cls(
            data=load_dataset(dataset_name, **kwargs),
            corpus=corpus,
            query_column=query_column,
            pos_target_column=pos_target_column,
            neg_target_column=neg_target_column,
        )

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

        return {
            self.QUERY_KEY: self.get_documents(sample_data, self.query_column),
            self.POS_TARGET_KEY: self.get_documents(sample_data, self.pos_target_column),
            self.NEG_TARGET_KEY: self.get_documents(sample_data, self.neg_target_column)
            if self.neg_target_column
            else None,
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

        if column.corpus_column is None or self.corpus is None:
            return doc_or_id
        else:
            # If the corpus is provided, retrieve the document using the corpus
            corpus_data = self.corpus.retrieve(doc_or_id)
            return Document(
                item=corpus_data[column.corpus_column],
                metadata={col: corpus_data[col] for col in corpus_data.keys() if col in column.corpus_metadata_columns},
            )


if __name__ == "__main__":
    from colpali_engine.data.corpus.local import LocalCorpus

    # corpus = LocalCorpus("/home/paulteiletche/VLM2Vec/data/MMEB/MMEB-train/images/VisDial/Train")
    # ds = IRDataset.from_hf(
    #     dataset_name="TIGER-Lab/MMEB-train",
    #     corpus=corpus,
    #     query_column=IRColumn("qry", desc_column=None),
    #     pos_target_column=IRColumn("pos_image_path", desc_column="pos_text", corpus_column="doc"),
    #     neg_target_column=IRColumn("neg_image_path", desc_column="neg_text", corpus_column="doc"),
    #     name="VisDial",
    #     split="original",
    # )

    corpus = LocalCorpus("/home/paulteiletche/VLM2Vec/data/MMEB/MMEB-train/images/MSCOCO_i2t/Train")
    ds = IRDataset.from_hf(
        dataset_name="TIGER-Lab/MMEB-train",
        corpus=corpus,
        query_column=IRColumn("qry_image_path", corpus_column="doc"),
        pos_target_column=IRColumn("pos_text"),
        neg_target_column=None,
        name="MSCOCO_i2t",
        split="original",
    )

    print(ds[0])
