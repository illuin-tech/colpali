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
        add_suffix: bool = True,
        image_dataset: Optional[Dataset] = None,
    ):
        super().__init__(
            processor=processor,
            max_length=max_length,
            add_suffix=add_suffix,
        )
        if image_dataset is None:
            raise ValueError("`image_dataset` must be provided")
        self.image_dataset = cast(Dataset, image_dataset)

    def get_image_from_image_dataset(self, image_idx):
        return self.image_dataset[int(image_idx)]["image"]

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        tmp_examples = examples
        examples = []

        for example in tmp_examples:
            pos_image = self.get_image_from_image_dataset(example["gold_index"])
            pos_query = example["query"]

            # Randomly sample a negative image amongst the top 10
            neg_image = self.get_image_from_image_dataset(example["negs"][randint(0, 9)])

            examples += [{"image": pos_image, "query": pos_query, "neg_image": neg_image}]

        return self(examples)
