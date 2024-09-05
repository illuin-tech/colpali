from typing import List, Optional, cast

from PIL.Image import Image
from transformers import PreTrainedTokenizer, ProcessorMixin
from transformers.models.idefics2 import Idefics2Processor
from transformers.models.paligemma import PaliGemmaProcessor

from colpali_engine.utils.processing_utils.colidefics_processing_utils import (
    process_images_idefics,
    process_queries_idefics,
)
from colpali_engine.utils.processing_utils.colpali_processing_utils import (
    process_images_colpali,
    process_queries_colpali,
)


class CustomCollator:
    """
    Collator for training ColBERT-based vision retrieval models (e.g. ColPali).
    """

    def __init__(
        self,
        processor: Optional[ProcessorMixin] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 2048,
        add_suffix: bool = True,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_token_id = None
        self.max_length = max_length
        self.suffix = ""

        if tokenizer is None and processor is None:
            raise ValueError("Either processor or tokenizer should be provided.")

        if self.processor is not None:
            self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
                self.processor.tokenizer.additional_special_tokens.index("<image>")
            ]

            if self.tokenizer is not None:
                raise ValueError("Only one of processor or tokenizer should be provided.")

        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.processor.__class__.__name__ == "PaliGemmaProcessor":
            if self.processor.tokenizer.padding_side != "right":
                print("Setting padding side to right")
                self.processor.tokenizer.padding_side = "right"

        if add_suffix:
            if self.tokenizer:
                self.suffix = self.tokenizer.pad_token * 10
            else:
                self.suffix = self.processor.tokenizer.pad_token * 10

    def __call__(self, examples):
        if self.processor is None:
            return self.forward_text(examples)
        elif self.processor.__class__.__name__ == "Idefics2Processor":
            return self.forward_vision_idefics(examples)
        elif self.processor.__class__.__name__ == "PaliGemmaProcessor":
            return self.forward_vision_pali(examples)
        else:
            raise ValueError("Processor not supported.")

    def forward_text(self, examples):
        """
        Collate function for text-only examples.
        """
        # Placeholders
        texts_doc: List[str] = []
        texts_query: List[str] = []

        # Process each example
        for example in examples:
            text_query = example["query"] + self.suffix
            text_doc = example["doc"]

            texts_doc.append(text_doc.strip())
            texts_query.append(text_query.strip())

        if self.tokenizer is None:
            raise ValueError("Tokenizer should be provided for text collator.")

        # Process the documents
        batch_doc = self.tokenizer(
            texts_doc,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        # Process the queries
        batch_query = self.tokenizer(
            texts_query,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        # Prefix each key with "doc_" or "query_" to avoid key conflicts
        batch_all = {f"doc_{k}": v for k, v in batch_doc.items()}
        del batch_doc
        batch_query = {f"query_{k}": v for k, v in batch_query.items()}
        batch_all.update(batch_query)

        return batch_all

    def forward_vision_idefics(self, examples):
        """
        Collate function for ColIdefics2.
        """
        # Placeholders
        queries: List[str] | List[None] | List[str | None] = []  # some documents don't have a query
        images: List[Image] = []

        if self.processor is None or not isinstance(self.processor, Idefics2Processor):
            raise ValueError("Processor should be provided for vision collator.")

        # Process each example
        for example in examples:
            if example["query"] is not None:
                queries.append(cast(str, example["query"]))
            else:
                queries.append(None)

            image = cast(Image, example["image"])
            images.append(image)

        # Process the documents
        batch_doc = process_images_idefics(
            processor=self.processor,
            images=images,
        )

        # Process the queries
        batch_query = None

        # Check if some but not all queries are `None`
        if all([t is None for t in queries]):
            print("All queries are None. Returning None for all queries.")
        elif any([t is None for t in queries]):
            raise ValueError("Some queries are None. This collator does not support `None` queries yet.")
        else:
            queries = cast(List[str], queries)
            batch_query = process_queries_idefics(
                processor=self.processor,
                queries=queries,
                max_length=self.max_length,
                suffix=self.suffix,
            )

        # Prefix each key with "doc_" or "query_" to avoid key conflicts
        batch_all = {f"doc_{k}": v for k, v in batch_doc.items()}
        del batch_doc
        if batch_query is not None:
            batch_query = {f"query_{k}": v for k, v in batch_query.items()}
            batch_all.update(batch_query)
            del batch_query

        return batch_all

    def forward_vision_pali(self, examples):
        """
        Collate function for ColPaLi.
        """
        # Placeholders
        texts_query: List[str] | List[None] | List[str | None] = []  # some documents don't have a query
        images: List[Image] = []
        neg_images: List[Image] = []

        if self.processor is None or not isinstance(self.processor, PaliGemmaProcessor):
            raise ValueError("Processor should be provided for vision collator.")

        # Process each example
        for example in examples:
            if example["image"] is None:
                raise ValueError("Image is None - This collator does not support None images yet.")

            images.append(cast(Image, example["image"]))

            if "neg_image" in example and example["neg_image"] is not None:
                neg_images.append(cast(Image, example["neg_image"]))

            if example["query"] is None:
                texts_query.append(None)

        # Process the documents
        batch_doc = process_images_colpali(
            processor=self.processor,
            images=images,
        )

        # Process the negative documents (if available)
        batch_neg_doc = None
        if len(neg_images) > 0:
            batch_neg_doc = process_images_colpali(
                processor=self.processor,
                images=neg_images,
            )

        # Process the queries
        batch_query = None

        if all([t is None for t in texts_query]):
            print("All queries are `None`. Returning `None` for all queries.")
        elif any([t is None for t in texts_query]):
            # If it's the first query that is not None but the rest are None, then it's hard negatives.
            raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            texts_query = cast(List[str], texts_query)
            batch_query = process_queries_colpali(
                processor=self.processor,
                queries=texts_query,
                max_length=self.max_length,
                suffix=self.suffix,
            )

        # Prefix each key with "doc_" or "query_" to avoid key conflicts
        batch_all = {f"doc_{k}": v for k, v in batch_doc.items()}
        del batch_doc
        if batch_query is not None:
            batch_query = {f"query_{k}": v for k, v in batch_query.items()}
            batch_all.update(batch_query)
            del batch_query
        if batch_neg_doc is not None:
            batch_neg_doc = {f"neg_doc_{k}": v for k, v in batch_neg_doc.items()}
            batch_all.update(batch_neg_doc)

        return batch_all
