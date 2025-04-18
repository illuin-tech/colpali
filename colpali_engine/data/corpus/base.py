from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset
from tqdm import tqdm


class BaseCorpus:
    """
    Base class for handling corpus.
    """

    @abstractmethod
    def retrieve(self, docid: Any) -> Dict[str, Any]:
        """
        Get the corpus row from the given Doc ID.
        THIS FUNCTION MUST BE OVERWRITTEN IN THE SUBCLASS FOR CUSTOM DATASETS.

        Args:
            docid (str): The id of the document.

        Returns:
            Dict[str, Any]: The row corresponding to the Doc ID.
        """
        pass


class SimpleCorpus(BaseCorpus):
    """
    Corpus class for handling retrieving with simple mapping.

    Args:
        corpus_data (List[Dict[str, Any]]): List of dictionaries containing doc data.
        doc_column (str): The key in the dictionary that contains the doc data.
        id_column (Optional[str]): Doc ID's entry key. Defaults to None for corpus index as the ID.
    """

    def __init__(self, corpus_data: List[Dict[str, Any]], id_column: Optional[str] = None):
        """
        Initialize the corpus with the provided data.
        """
        self.corpus_data = corpus_data
        self.id_column = id_column
        self.id_to_idx = self._build_id_to_idx()

    @classmethod
    def from_hf(
        cls,
        corpus_name_or_path: Union[str, List[Dict[str, Any]]],
        id_column: Optional[str] = None,
        **kwargs,
    ) -> "SimpleCorpus":
        """
        Load the corpus from a Hugging Face dataset.

        Args:
            corpus_data_or_path (Union[str, List[Dict[str, Any]]]): Path to the dataset or list of dictionaries.
            doc_column (str): The key in the dictionary that contains the doc data.
            id_column (Optional[str]): The key in the dictionary that contains the doc ID. Defaults to None.

        Returns:
            SimpleCorpus: An instance of the SimpleCorpus class.
        """
        return cls(
            corpus_data=load_dataset(corpus_name_or_path, **kwargs),
            id_column=id_column,
        )

    def __len__(self) -> int:
        """
        Return the number of docs in the corpus.

        Returns:
            int: The number of docs in the corpus.
        """
        return len(self.corpus_data)

    def _build_id_to_idx(self) -> Dict[str, int]:
        """
        Build a mapping from doc IDs to indices in the corpus data.
        This is useful for retrieving docs by their IDs.

        Returns:
            Dict[str, int]: A dictionary mapping doc IDs to indices.
        """
        if self.id_column is None:
            print("No corpus ID column provided. Using index as ID.")
            return {idx: idx for idx in range(len(self.corpus_data))}

        return {
            doc[self.id_column]: idx
            for idx, doc in tqdm(
                enumerate(self.corpus_data),
                desc="Building ID to Index mapping",
                total=len(self.corpus_data),
            )
        }

    def retrieve(self, docid: Any) -> Dict[str, Any]:
        """
        Get the corpus row from the given Doc ID.

        Args:
            docid (str): The id of the document.

        Returns:
            Dict[str, Any]: The row corresponding to the Doc ID.
        """
        corpus_idx = self.id_to_idx.get(docid)
        if corpus_idx is None:
            raise ValueError(f"Document ID {docid} not found in the corpus.")
        return self.corpus_data[corpus_idx]
