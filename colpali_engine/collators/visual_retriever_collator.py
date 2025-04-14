import random
from typing import Any, Dict, List, Union

from PIL.Image import Image

from colpali_engine.data.dataset import Document, IRDataset
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

    # Input keys
    query_key = IRDataset.QUERY_KEY
    pos_target_key = IRDataset.POS_TARGET_KEY
    neg_target_key = IRDataset.NEG_TARGET_KEY
    # Prefixes
    query_prefix = "query_"
    pos_doc_prefix = "doc_"
    neg_doc_prefix = "neg_doc_"

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_token_id = None

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
        queries: List[Union[None, str, Image]] = []
        pos_targets: List[Union[str, Image]] = []
        neg_targets: List[Union[str, Image]] = []

        # Parse the examples.
        for example in examples:
            query = example.get(self.query_key)
            sampled_query = random.choice(query) if isinstance(query, list) else query
            queries.append(sampled_query)

            pos_tgt = example.get(self.pos_target_key)
            if pos_tgt is not None:
                sample_pos = random.choice(pos_tgt) if isinstance(pos_tgt, list) else pos_tgt
                pos_targets.append(sample_pos)
            else:
                raise ValueError("Image is None - This collator does not support None images yet.")

            neg_tgt = example.get(self.neg_target_key)
            if neg_tgt is not None:
                sampled_neg = random.choice(neg_tgt) if isinstance(neg_tgt, list) else neg_tgt
                neg_targets.append(sampled_neg)

        # Process queries.
        if all(q is None for q in queries):
            batch_query = None
        elif any(q is None for q in queries):
            raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            batch_query = self.auto_collate(queries, prefix=self.query_prefix)

        # Process targets.
        batch_pos_target = self.auto_collate(pos_targets, prefix=self.pos_doc_prefix)
        batch_neg_target = self.auto_collate(neg_targets, prefix=self.neg_doc_prefix) if neg_targets else {}

        return {
            **batch_query,
            **batch_pos_target,
            **batch_neg_target,
        }

    def auto_collate(self, batch: List[Document], prefix: str = "") -> Dict[str, Any]:
        """Automatically collate a batch of documents."""
        # Convert Document objects to their underlying data.
        batch = [b.item if isinstance(b, Document) else b for b in batch]
        if isinstance(batch[0], str):
            return self.collate_texts(batch, prefix=prefix)
        elif isinstance(batch[0], Image):
            return self.collate_images(batch, prefix=prefix)
        else:
            raise ValueError(f"Unsupported batch type: {type(batch[0])}. Expected str or Image.")

    def collate_images(self, images: List[Image], prefix: str = "") -> Dict[str, Any]:
        """Collate images into a batch."""
        # Process images.
        batch_im = self.processor.process_images(images=images)
        # Prefix keys to avoid collisions.
        return prefix_keys(batch_im, prefix)

    def collate_texts(self, texts: List[str], prefix: str = "") -> Dict[str, Any]:
        """Collate texts into a batch."""
        # Process texts.
        batch_text = self.processor.process_queries(
            queries=texts,
            max_length=self.max_length,
        )
        # Prefix keys to avoid collisions.
        return prefix_keys(batch_text, prefix)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from colpali_engine.data.corpus.local import LocalCorpus
    from colpali_engine.data.dataset import IRColumn, IRDataset
    from colpali_engine.models.idefics3 import ColIdefics3Processor

    # corpus = LocalCorpus("/home/paulteiletche/VLM2Vec/data/MMEB/MMEB-train/images/VisDial/Train")
    # ds = IRDataset.from_hf(
    #     dataset_name="TIGER-Lab/MMEB-train",
    #     corpus=corpus,
    #     query_column=IRColumn("qry", desc_column=None),
    #     pos_target_column=IRColumn("pos_image_path", desc_column="pos_text", corpus_column="doc"),
    #     neg_target_column=IRColumn("neg_image_path", desc_column="neg_text", corpus_column="doc"),
    #     name="VisDial",
    #     split="original",
    # )

    corpus = LocalCorpus("/home/paulteiletche/VLM2Vec/data/MMEB/MMEB-train/images/MSCOCO_i2t/Train")
    ds = IRDataset.from_hf(
        dataset_name="TIGER-Lab/MMEB-train",
        corpus=corpus,
        query_column=IRColumn("qry_image_path", corpus_column="doc"),
        pos_target_column=IRColumn("pos_text"),
        neg_target_column=None,
        name="MSCOCO_i2t",
        split="original",
    )

    collator = VisualRetrieverCollator(
        processor=ColIdefics3Processor.from_pretrained("vidore/colSmol-256M"),
        max_length=2048,
    )

    dataloader = DataLoader(
        ds,
        batch_size=2,
        collate_fn=collator,
    )

    for batch in dataloader:
        print("First Batch:")
        print(f"Input keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['query_input_ids'].shape}")
        print(f"Attention mask shape: {batch['query_attention_mask'].shape}")
        if "query_pixel_values" in batch:
            print(f"Query pixel values shape: {batch['query_pixel_values'].shape}")
        if "doc_pixel_values" in batch:
            print(f"Pixel values shape: {batch['doc_pixel_values'].shape}")
        if "neg_doc_pixel_values" in batch:
            print(f"Negative pixel values shape: {batch['neg_doc_pixel_values'].shape}")
        break
