from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image
from transformers import BatchEncoding, BatchFeature


class BaseVisualRetrieverProcessor(ABC):
    """
    Base class for visual retriever processors.
    """

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
