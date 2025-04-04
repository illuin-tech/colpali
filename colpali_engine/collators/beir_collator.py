import random
from typing import Any, Dict, List, Optional

from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class BEIRCollator(VisualRetrieverCollator):
    """
    Collator for BEIR-style dataset training.
    """

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        use_translations: bool = False,
        corpus: Optional["Dataset"] = None,  # noqa: F821
    ):
        super().__init__(processor=processor, max_length=max_length)
        if corpus is None:
            raise ValueError("Corpus is required for BEIRCollator")
        self.use_translations = use_translations
        self.corpus = corpus

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        processed_examples: List[Dict[str, Any]] = []
        for example in examples:
            # Extract positive document id depending on the corpus format.
            positive_passages = example["positive_passages"]
            if example.get("answerabilities") is not None:
                answerabilities = example["answerabilities"]
                positive_passages = [
                    passage_id
                    for (passage_id, answerability) in zip(example["positive_passages"], answerabilities)
                    if answerability > 0
                ]
            else:
                positive_passages = example["positive_passages"]

            if example.get("original_query") is not None:
                if self.use_translations:
                    query = random.choice([example["original_query"]] + example["translated_queries"])
                else:
                    query = example["original_query"]
            else:
                query = example["query"]

            pos_docid = random.choice(positive_passages)

            sample = {
                "image": self.corpus[pos_docid]["image"],
                "query": query,
                "image_id": pos_docid,
                "positive_ids": positive_passages,
            }

            processed_examples.append(sample)

        return super().__call__(processed_examples)
