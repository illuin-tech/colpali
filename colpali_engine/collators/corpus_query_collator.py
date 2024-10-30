from random import randint
from typing import Any, Dict, List, Optional, cast

from datasets import Dataset

from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class CorpusQueryCollator(VisualRetrieverCollator):
    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        add_suffix: bool = True,
        image_dataset: Optional[Dataset] = None,
        mined_negatives: bool = True,
    ):
        super().__init__(
            processor=processor,
            max_length=max_length,
            add_suffix=add_suffix,
        )
        if image_dataset is None:
            raise ValueError("`image_dataset` must be provided")
        self.image_dataset = cast(Dataset, image_dataset)
        self.mined_negatives = mined_negatives

    def get_image_from_image_dataset(self, image_idx):
        return self.image_dataset[int(image_idx)]["image"]

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        tmp_examples = examples
        examples = []

        for example in tmp_examples:
            pos_image = self.get_image_from_image_dataset(example["positive_passages"][0]["doc_id"])
            pos_query = example["query"]
            sample = {"image": pos_image, "query": pos_query}
            if self.mined_negatives:
                # Randomly sample a negative image
                len_negs = len(example["negative_passages"])
                neg_image = self.get_image_from_image_dataset(example["negative_passages"][randint(0, len_negs - 1)])
                sample.update({"neg_image": neg_image})
            examples += [sample]

        return super().__call__(examples)
