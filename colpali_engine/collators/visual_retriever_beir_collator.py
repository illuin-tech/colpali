from typing import Any, Dict, List, Optional

from colpali_engine.collators.visual_retriever_qa_collator import VisualRetrieverQACollator
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class VisualRetrieverBEIRCollator(VisualRetrieverQACollator):
    """
    Collator for training vision retrieval models with a BEIR dataset.
    """

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        ds_queries: Optional["Dataset"] = None,  # noqa: F821
        ds_passages: Optional["Dataset"] = None,  # noqa: F821
    ):
        super().__init__(processor=processor, max_length=max_length)
        raise NotImplementedError

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError
