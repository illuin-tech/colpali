import os
from typing import Any, Dict, Optional

from PIL import Image

from colpali_engine.data.corpus.base import BaseCorpus


class LocalCorpus(BaseCorpus):
    """
    Dataset class for the local corpus.
    """

    def __init__(self, corpus_path: str, doc_column_name: str = "doc", file_type: Optional[str] = "image"):
        self.corpus_path = corpus_path
        self.doc_column_name = doc_column_name
        self.file_type = file_type

    def __len__(self) -> int:
        return len(os.listdir(self.corpus_path))

    def path2doc(self, path: str) -> Any:
        """
        Convert a path to a document.
        Args:
            path (str): The path to the document.
        Returns:
            Doc: The document corresponding to the path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        if self.file_type == "image":
            return Image.open(path)
        elif self.file_type == "text":
            with open(path, "r") as f:
                return f.read()
        else:
            raise ValueError("Unsupported file type. Supported types are 'image' and 'text'.")

    def retrieve(self, docid: Any) -> Dict[str, Any]:
        """
        Get the corpus row from the given Doc ID.

        Args:
            docid (str): The id of the document.

        Returns:
            Dict[str, Any]: The row corresponding to the Doc ID.
        """
        path = os.path.join(self.corpus_path, os.path.basename(docid))
        return {self.doc_column_name: self.path2doc(path)}
