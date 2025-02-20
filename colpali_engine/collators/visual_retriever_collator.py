from typing import Any, Dict, List, Union, cast

from colpali_engine.models.idefics_2 import ColIdefics2Processor
from colpali_engine.models.paligemma import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class VisualRetrieverCollator:
    """
    Collator for training vision retrieval models.
    """

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.image_token_id = None
        self.max_length = max_length

        if isinstance(self.processor, ColPaliProcessor) or isinstance(self.processor, ColIdefics2Processor):
            self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
                self.processor.tokenizer.additional_special_tokens.index("<image>")
            ]

        if isinstance(self.processor, ColPaliProcessor):
            if self.processor.tokenizer.padding_side != "right":
                print("Setting padding side to right")
                self.processor.tokenizer.padding_side = "right"


    def __call__(
        self,
        examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Collate function for the vision retriever associated to the collator's processor.
        """
        # if self.processor is None or not isinstance(self.processor, BaseVisualRetrieverProcessor):
        #     raise ValueError("Processor should be provided for vision collator.")

        # Process the documents
        breakpoint()
        batch_doc = self.processor.process_images(
            images=examples["image"]
        )

        # Process the negative documents (if available)
        batch_neg_doc = None
        if "neg_image" in examples:
            batch_neg_doc = self.processor.process_images(
                images=examples["neg_image"]
            )

        # Process the queries
        batch_query = None

        if all([t is None for t in texts_query]):
            # print("All queries are `None`. Returning `None` for all queries.")
            pass
        elif any([t is None for t in texts_query]):
            # If it's the first query that is not None but the rest are None, then it's hard negatives.
            raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            batch_query = self.processor.process_queries(
                queries=examples["query"],
                max_length=self.max_length,
            )

        # Prefix each key with "doc_" or "query_" to avoid key conflicts
        batch_all = {f"doc_{k}": v for k, v in batch_doc.items()}

        if batch_query is not None:
            batch_all.update({f"query_{k}": v for k, v in batch_query.items()})

        if batch_neg_doc is not None:
            batch_all.update({f"neg_doc_{k}": v for k, v in batch_neg_doc.items()})

        return batch_all
