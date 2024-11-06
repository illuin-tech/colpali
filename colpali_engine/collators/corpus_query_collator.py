from __future__ import annotations

from random import randint
from typing import Any, Dict, List, Optional

from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class CorpusQueryCollator(VisualRetrieverCollator):
    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        image_dataset: Optional["Dataset"] = None, # noqa: F821
        mined_negatives: bool = True,
        corpus_format: str = "wikiss",
    ):
        super().__init__(
            processor=processor,
            max_length=max_length,
        )
        if image_dataset is None:
            raise ValueError("`image_dataset` must be provided")
        self.image_dataset = image_dataset
        self.mined_negatives = mined_negatives
        self.corpus_format = corpus_format

        if self.corpus_format == "wikiss":
            print("Mapping docids to indices")
            self.docid_to_idx = {docid: idx for docid, idx in
                                 zip(self.image_dataset["docid"], range(len(self.image_dataset)))}

    def get_image_from_docid(self, docid):
        if self.corpus_format == "wikiss":
            return self.image_dataset[self.docid_to_idx[docid]]["image"]
        elif self.corpus_format == "docmatix":
            doc_page = int(docid.split("_")[1])
            doc_id = int(docid.split("_")[0])
            return self.image_dataset[doc_id]["images"][doc_page]
        elif self.corpus_format == "vidore":
            return self.image_dataset[int(docid)]["image"]
        else:
            raise NotImplementedError(f"Corpus format {self.corpus_format} not supported")


    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        tmp_examples = examples
        examples = []

        for example in tmp_examples:
            if self.corpus_format == "wikiss" or self.corpus_format == "docmatix":
                pos_image = self.get_image_from_docid(example["positive_passages"][0]["docid"])
                pos_query = example["query"]
                sample = {"image": pos_image, "query": pos_query}
                if self.mined_negatives:
                    # Randomly sample a negative image
                    len_negs = len(example["negative_passages"])
                    neg_image = self.get_image_from_docid(
                        example["negative_passages"][randint(0, len_negs - 1)]["docid"])
                    sample.update({"neg_image": neg_image})
                examples += [sample]

            elif self.corpus_format == "vidore":
                pos_image = self.get_image_from_docid(example["positive_passages"][0])
                pos_query = example["query"]
                sample = {"image": pos_image, "query": pos_query}
                if self.mined_negatives:
                    # Randomly sample a negative image
                    len_negs = len(example["negative_passages"])
                    neg_image = self.get_image_from_docid(example["negative_passages"][randint(0, len_negs - 1)])
                    sample.update({"neg_image": neg_image})
                examples += [sample]
            else:
                raise NotImplementedError(f"Corpus format {self.corpus_format} not supported")


        return super().__call__(examples)
