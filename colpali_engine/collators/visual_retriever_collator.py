from typing import Any, Dict, List, Union

import torch
from PIL.Image import Image

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
        process_images_before_training: bool = False,
    ):
        self.processor = processor
        self.image_token_id = None
        self.max_length = max_length
        self.process_images_before_training = process_images_before_training

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
        if self.process_images_before_training:
            return self.offline_processing(examples)
        return self.online_processing(examples)


    def online_processing(
        self,
        examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Collate function for the vision retriever associated to the collator's processor.
        """
        # Placeholders
        texts_query: Union[List[str], List[None], List[Union[str, None]]] = []  # some documents don't have a query
        images: List[Image] = []
        neg_images: List[Image] = []
        # breakpoint()


        # Process each example
        for example in examples:
            texts_query.append(example["query"])
            images.append(example["image"])

            if "neg_image" in example and example["neg_image"] is not None:
                neg_images.append(example["neg_image"])

        # Process the documents
        batch_doc = self.processor.process_images(
            images=images,
        )

        # Process the negative documents (if available)
        batch_neg_doc = None
        if len(neg_images) > 0:
            batch_neg_doc = self.processor.process_images(
                images=neg_images,
            )

        # Process the queries
        batch_query = None

        if all([t is None for t in texts_query]):
            # print("All queries are `None`. Returning `None` for all queries.")
            pass
        else:
            texts_query: List[str] = texts_query
            batch_query = self.processor.process_queries(
                queries=texts_query,
                max_length=self.max_length,
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


    def offline_processing(
        self,
        examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Collate function for the vision retriever associated to the collator's processor.
        """
        # Placeholders
        texts_query = []
        pixel_values = []
        image_grid_thw = []
        input_ids = []
        attention_mask = []
        neg_pixel_values = []
        neg_image_grid_thw = []
        neg_input_ids = []
        neg_attention_mask = []

        breakpoint()

        for example in examples:
            texts_query.append(example["query"])
            pixel_values.append(example["pixel_values"])
            image_grid_thw.append(example["image_grid_thw"])
            input_ids.append(example["input_ids"])
            attention_mask.append(example["attention_mask"])

            if "neg_pixel_values" in example:
                neg_pixel_values.append(example["neg_pixel_values"])
                neg_image_grid_thw.append(example["neg_image_grid_thw"])
                neg_input_ids.append(example["neg_input_ids"])
                neg_attention_mask.append(example["neg_attention_mask"])

        # Pad pixel values
        pixel_values = torch.nn.utils.rnn.pad_sequence(pixel_values, batch_first=True, padding_value=0)
        image_grid_thw = torch.stack(image_grid_thw)

        # Pad input sequences
        batch_doc = self.processor.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt"
        )

        batch_all = {
            "doc_pixel_values": pixel_values,
            "doc_image_grid_thw": image_grid_thw,
            "doc_input_ids": batch_doc["input_ids"],
            "doc_attention_mask": batch_doc["attention_mask"],
        }

        # Process queries
        if any(texts_query):  # Ensure there are valid queries
            batch_query = self.processor.process_queries(
                queries=texts_query,
                max_length=self.max_length
            )
            batch_all["query_input_ids"] = batch_query["input_ids"]
            batch_all["query_attention_mask"] = batch_query["attention_mask"]

        # Process negatives if present
        if neg_pixel_values:
            neg_pixel_values = torch.nn.utils.rnn.pad_sequence(neg_pixel_values, batch_first=True, padding_value=0)
            neg_image_grid_thw = torch.stack(neg_image_grid_thw)

            batch_neg_doc = self.processor.tokenizer.pad(
                {"input_ids": neg_input_ids, "attention_mask": neg_attention_mask},
                padding=True,
                return_tensors="pt"
            )

            batch_all.update({
                "neg_doc_pixel_values": neg_pixel_values,
                "neg_doc_image_grid_thw": neg_image_grid_thw,
                "neg_doc_input_ids": batch_neg_doc["input_ids"],
                "neg_doc_attention_mask": batch_neg_doc["attention_mask"],
            })

        return batch_all
