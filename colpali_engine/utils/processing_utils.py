from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image
from transformers import BatchEncoding, BatchFeature, PreTrainedTokenizer, ProcessorMixin


class BaseVisualRetrieverProcessor(ABC, ProcessorMixin):
    """
    Base class for visual retriever processors.
    """

    def __init__(self):
        if not hasattr(self, "tokenizer"):
            raise ValueError("Processor must have a tokenizer attribute.")

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizer:
        pass

    @abstractmethod
    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature | BatchEncoding:
        pass

    @abstractmethod
    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature | BatchEncoding:
        pass