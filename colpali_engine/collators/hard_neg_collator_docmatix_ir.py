from typing import Any, Dict, List, Optional, cast

from datasets import Dataset

from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class HardNegDocmatixCollator(VisualRetrieverCollator):
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

    def get_image_from_docid(self, docid):
        example_idx, image_idx = docid.split("_")
        target_image = self.image_dataset[int(example_idx)]["images"][int(image_idx)]
        return target_image

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        tmp_examples = examples
        examples = []
        for example in tmp_examples:
            pos_image = self.get_image_from_docid(example["positive_passages"][0]["docid"])
            pos_query = example["query"]
            neg_images_ids = [doc["docid"] for doc in example["negative_passages"][:1]]
            neg_images = [self.get_image_from_docid(docid) for docid in neg_images_ids]

            examples += [{"image": pos_image, "query": pos_query, "neg_image": neg_images[0]}]

        return self(examples)
