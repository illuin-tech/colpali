import random
from typing import Any, Dict, List, Union

from PIL.Image import Image

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.models.paligemma import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


def prefix_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Prefix all keys in a dictionary with the given prefix.
    """
    return {f"{prefix}{k}": v for k, v in data.items()}


class VisualRetrieverCollator:
    """
    Collator for training vision retrieval models.
    """

    # Input keys
    query_key = ColPaliEngineDataset.QUERY_KEY
    pos_target_key = ColPaliEngineDataset.POS_TARGET_KEY
    neg_target_key = ColPaliEngineDataset.NEG_TARGET_KEY
    # Prefixes
    query_prefix = "query_"
    pos_doc_prefix = "doc_"
    neg_doc_prefix = "neg_doc_"

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_token_id = None

        # If processor is one of the supported types, extract the <image> token id.
        if isinstance(self.processor, (ColPaliProcessor,)):
            image_token = "<image>"
            try:
                idx = self.processor.tokenizer.additional_special_tokens.index(image_token)
                self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[idx]
            except ValueError:
                self.image_token_id = None

        # Force padding to be on the right for ColPaliProcessor.
        if isinstance(self.processor, ColPaliProcessor) and self.processor.tokenizer.padding_side != "right":
            print("Setting padding side to right")
            self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries: List[Union[None, str, Image]] = []
        pos_targets: List[Union[str, Image]] = []
        neg_targets: List[Union[str, Image]] = []

        # Parse the examples.
        for example in examples:
            query = example.get(self.query_key)
            sampled_query = random.choice(query) if isinstance(query, list) else query
            queries.append(sampled_query)

            pos_tgt = example.get(self.pos_target_key)
            if pos_tgt is not None:
                sample_pos = random.choice(pos_tgt) if isinstance(pos_tgt, list) else pos_tgt
                pos_targets.append(sample_pos)
            else:
                raise ValueError("Image is None - This collator does not support None images yet.")

            neg_tgt = example.get(self.neg_target_key)
            if neg_tgt is not None:
                sampled_neg = random.choice(neg_tgt) if isinstance(neg_tgt, list) else neg_tgt
                neg_targets.append(sampled_neg)

        # Process queries.
        if all(q is None for q in queries):
            batch_query = None
        elif any(q is None for q in queries):
            raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            batch_query = self.auto_collate(queries, prefix=self.query_prefix)

        # Process targets.
        batch_pos_target = self.auto_collate(pos_targets, prefix=self.pos_doc_prefix)
        batch_neg_target = self.auto_collate(neg_targets, prefix=self.neg_doc_prefix) if neg_targets else {}

        return {
            **batch_query,
            **batch_pos_target,
            **batch_neg_target,
        }

    def auto_collate(self, batch: List[Union[str, Image]], prefix: str = "") -> Dict[str, Any]:
        """Automatically collate a batch of documents."""
        # Convert Document objects to their underlying data.
        if isinstance(batch[0], str):
            return self.collate_texts(batch, prefix=prefix)
        elif isinstance(batch[0], Image):
            return self.collate_images(batch, prefix=prefix)
        else:
            raise ValueError(f"Unsupported batch type: {type(batch[0])}. Expected str or Image.")

    def collate_images(self, images: List[Image], prefix: str = "") -> Dict[str, Any]:
        """Collate images into a batch."""
        # Process images.
        batch_im = self.processor.process_images(images=images)
        # Prefix keys to avoid collisions.
        return prefix_keys(batch_im, prefix)

    def collate_texts(self, texts: List[str], prefix: str = "") -> Dict[str, Any]:
        """Collate texts into a batch."""
        # Process texts.
        batch_text = self.processor.process_queries(
            queries=texts,
            max_length=self.max_length,
        )
        # Prefix keys to avoid collisions.
        return prefix_keys(batch_text, prefix)
