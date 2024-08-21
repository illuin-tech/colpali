from typing import Any, Dict, List

import torch

from colpali_engine.models.colpali_2.colpali_2_processor import ColPali2Processor


class ColPali2Collator:
    def __init__(
        self,
        processor: ColPali2Processor,
        max_length: int = 2048,
        add_suffix: bool = False,
    ):
        self.processor = processor
        self.image_token_id = None
        self.max_length = max_length
        self.suffix = ""
        if add_suffix:
            self.suffix = "\n" * 10

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Placeholders
        texts_query = []
        images = []

        # Populate the placeholders
        for example in examples:
            if example["image"] is None:
                raise ValueError("Image is None - This collator does not support `None` images yet.")

            image = example["image"].convert("RGB")
            images.append(image)

            if example["query"] is None:
                texts_query.append(None)
            else:
                query = example["query"]
                query = f"Question: {query}"
                texts_query.append(query)

        # Process the documents
        batch_doc = self.processor.process_image(
            image=images,
            padding="longest",
            do_convert_rgb=True,
            return_tensors="pt",
            add_instruction_prompt=True,
        )

        # Process the queries
        batch_query = None

        # Check if some but not all queries are `None`
        if all([t is None for t in texts_query]):
            print("All queries are None. Returning `None` for all queries.")
        elif any([t is None for t in texts_query]):
            raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            batch_query = self.processor.process_image(
                image=images,
                padding="longest",
                do_convert_rgb=True,
                return_tensors="pt",
                add_instruction_prompt=True,
            )

        # Prefix each key in ouptut dict with "doc_" or "query_" to avoid key conflicts
        batch_all = {f"doc_{k}": v for k, v in batch_doc.items()}
        del batch_doc
        if batch_query is not None:
            batch_query = {f"query_{k}": v for k, v in batch_query.items()}
            batch_all.update(batch_query)
            del batch_query

        return batch_all
