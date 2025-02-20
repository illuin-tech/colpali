import torch

from typing import Any, Dict, List, Union, cast

from transformers.image_processing_base import BatchFeature

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
        minibatch_size: int = 32,
    ):
        self.processor = processor
        self.image_token_id = None
        self.max_length = max_length
        self.minibatch_size = minibatch_size

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
        # Placeholders
        texts_query: Union[List[str], List[None], List[Union[str, None]]] = []  # some documents don't have a query

        # if self.processor is None or not isinstance(self.processor, BaseVisualRetrieverProcessor):
        #     raise ValueError("Processor should be provided for vision collator.")

        # Process each example
        tmp_batch_doc = []
        tmp_batch_neg_doc = []
        for i in range(0, len(examples), self.minibatch_size):
            # Process the documents
            breakpoint()
            tmp_batch_doc += [self.processor.process_images(
                images=examples[i : i + self.minibatch_size]["image"],
            )]

            # Process the negative documents (if available)
            batch_neg_doc = None
            if "neg_image" in examples[i]:
                tmp_batch_neg_doc += [self.processor.process_images(
                    images=examples[i : i + self.minibatch_size]["neg_image"],
                )]

        batch_doc = {}
        batch_neg_doc = None if tmp_batch_neg_doc is None else {}
        for key in tmp_batch_doc[0].keys():
            batch_doc[key] = torch.nn.utils.rnn.pad_sequence([a for b in tmp_batch_doc for a in b[key]], batch_first=True, padding_value=0)

        batch_doc = BatchFeature(batch_doc)

        if tmp_batch_neg_doc is not None:
            for key in tmp_batch_neg_doc[0].keys():
                batch_neg_doc[key] = torch.nn.utils.rnn.pad_sequence([a for b in tmp_batch_neg_doc for a in b[key]], batch_first=True, padding_value=0)

            batch_neg_doc = BatchFeature(batch_neg_doc)

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


if __name__ == "__main__":
    from PIL import Image

    processor = ColPaliProcessor.from_pretrained("vidore/colpali")
    collator = VisualRetrieverCollator(processor=processor, minibatch_size=2)
    examples = [
        {
            "image": Image.new("RGB", (100, 100)),
            "query": "What is this?",
        },
        {
            "image": Image.new("RGB", (150, 100)),
            "query": "What is this?",
        },
        {
            "image": Image.new("RGB", (200, 100)),
            "query": "What is this?",
        },
        {
            "image": Image.new("RGB", (100, 200)),
            "query": "What is this?",
        },
    ]
    from datasets import Dataset
    collator(Dataset.from_list(examples))