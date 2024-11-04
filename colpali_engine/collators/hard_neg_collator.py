from random import randint
from typing import Any, Dict, List, Optional, cast

from datasets import Dataset

from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class HardNegCollator(VisualRetrieverCollator):
    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        image_dataset: Optional[Dataset] = None,
    ):
        raise DeprecationWarning("Deprecated - use CorpusQueryCollator instead")
