from typing import Any, Dict, List, Union, cast

import torch
from PIL.Image import Image

from colpali_engine.models.idefics_2 import ColIdefics2Processor
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

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        deduplicate_images: bool = False,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_token_id = None
        self.deduplicate_images = deduplicate_images

        # If processor is one of the supported types, extract the <image> token id.
        if isinstance(self.processor, (ColPaliProcessor, ColIdefics2Processor)):
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
        texts_query: List[Union[str, None]] = []
        images: List[Image] = []
        neg_images: List[Image] = []
        labels: List[int] = []

        # Parse the examples.
        ids_in_position = []
        if self.deduplicate_images:
            id2position = {}
            counter = 0
            print("examples", examples)
            for example in examples:
                query = example.get("query")
                texts_query.append(query)
                if example["image_id"] not in id2position:
                    id2position[example["image_id"]] = counter
                    ids_in_position.append(example["image_id"])
                    counter += 1

                    image = example.get("image")
                    if image is None:
                        raise ValueError("Image is None - This collator does not support None images yet.")
                    images.append(cast(Image, image))

                neg_image = example.get("neg_image")
                if neg_image is not None:
                    neg_images.append(cast(Image, neg_image))

            labels: List[int] = [[0] * len(id2position)] * len(texts_query)
            for i, example in enumerate(examples):
                for id_ in ids_in_position:
                    if id_ in example["positive_ids"]:
                        labels[i][id2position[id_]] = 1
        else:
            for example in examples:
                query = example.get("query")
                texts_query.append(query)

                image = example.get("image")
                if image is None:
                    raise ValueError("Image is None - This collator does not support None images yet.")
                images.append(cast(Image, image))
                ids_in_position.append(example["image_id"])
                neg_image = example.get("neg_image")
                if neg_image is not None:
                    neg_images.append(cast(Image, neg_image))

            labels: List[int] = [[0] * len(images)] * len(texts_query)
            for i, example in enumerate(examples):
                for j, id_ in enumerate(ids_in_position):
                    if id_ in example["positive_ids"]:
                        labels[i][j] = 1
        # Process images.
        batch_doc = self.processor.process_images(images=images)
        batch_neg_doc = self.processor.process_images(images=neg_images) if neg_images else None
        batch_neg_doc = self.processor.process_images(images=neg_images) if neg_images else None

        # Process queries.
        if all(q is None for q in texts_query):
            batch_query = None
        elif any(q is None for q in texts_query):
            raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            batch_query = self.processor.process_queries(
                queries=cast(List[str], texts_query),
                max_length=self.max_length,
            )

        # Prefix keys to avoid collisions.
        batch_all = prefix_keys(batch_doc, "doc_")
        if batch_query:
            batch_all.update(prefix_keys(batch_query, "query_"))
        if batch_neg_doc:
            batch_all.update(prefix_keys(batch_neg_doc, "neg_doc_"))
        batch_all["labels"] = torch.Tensor(labels)

        return batch_all
