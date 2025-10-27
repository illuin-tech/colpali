import random
import torch
from typing import Any, Dict, List, Union

from PIL.Image import Image

from colpali_engine.data.dataset import ColPaliEngineDataset
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
        if isinstance(self.processor, (ColPaliProcessor,)):
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
        selected_ids: List[int] = []

        # Parse the examples.
        positive_ids_tensor = -torch.ones((len(examples), 100), dtype=torch.long)
        for i, example in enumerate(examples):
            assert ColPaliEngineDataset.QUERY_KEY in example, f"Missing {ColPaliEngineDataset.QUERY_KEY} in example."
            query = example[ColPaliEngineDataset.QUERY_KEY]
            sampled_query = random.choice(query) if isinstance(query, list) else query
            queries.append(sampled_query)

            assert ColPaliEngineDataset.POS_TARGET_KEY in example, (
                f"Missing {ColPaliEngineDataset.POS_TARGET_KEY} in example."
            )
            pos_tgt = example[ColPaliEngineDataset.POS_TARGET_KEY]
            positive_ids = example.get("positive_ids", None)
            if isinstance(pos_tgt, list):
                sample_tuple = random.choice([(t, id_) for t, id_ in zip(pos_tgt, positive_ids)])
                sample_pos = sample_tuple[0]
                selected_ids.append(sample_tuple[1])
            else:
                sample_pos = pos_tgt
            pos_targets.append(sample_pos)
            if positive_ids is not None:
                positive_ids_tensor[i, :len(positive_ids)] = torch.tensor(positive_ids)

            neg_tgt = example.get(ColPaliEngineDataset.NEG_TARGET_KEY, None)
            if neg_tgt is not None:
                # sampled_neg = random.choice(neg_tgt) if isinstance(neg_tgt, list) else neg_tgt
                # neg_targets.append(random.choice(neg_tgt)) #neg_tgts)
                neg_targets.append(neg_tgt)

        # Ensure all queries are strings or images.
        assert all(isinstance(q, str) for q in queries), (
            "All queries must be strings, this collator does not support images in queries."
        )

        # Process queries.
        queries = [self.processor.query_prefix + q + self.processor.query_augmentation_token * 10 for q in queries]
        batch_query = self.auto_collate(queries, key_prefix=self.query_prefix)

        # Process targets.
        batch_pos_target = self.auto_collate(pos_targets, key_prefix=self.pos_doc_prefix)
        batch_neg_target = self.auto_collate(neg_targets, key_prefix=self.neg_doc_prefix) if neg_targets else {}

        return {
            **batch_query,
            **batch_pos_target,
            **batch_neg_target,
            "selected_ids": torch.Tensor(selected_ids),
            "positive_ids_tensor": positive_ids_tensor,
        }

    def auto_collate(self, batch: List[Union[str, Image, List[str], List[Image]]], key_prefix: str = "") -> Dict[str, Any]:
        """Automatically collate a batch of documents."""
        # Convert Document objects to their underlying data.
        # if type is mixed across the batch, raise an error.
        all_types = set(type(item) for item in batch)
        if str in all_types and Image in all_types:
            raise ValueError(f"Batch contains mixed types: {all_types}. Expected all items to be of the same type.")
        if isinstance(batch[0], str):
            proc_batch = self.processor.process_texts(texts=batch)
        elif isinstance(batch[0], Image):
            proc_batch = self.processor.process_images(images=batch)
        elif isinstance(batch[0], list):
            if isinstance(batch[0][0], str):
                proc_texts_batch = []
                batch_size = len(batch)
                all_texts = [text for texts in batch for text in texts]
                num_negatives = len(all_texts) // batch_size
                proc_batch = self.processor.process_texts(texts=all_texts)
            elif isinstance(batch[0][0], Image):
                proc_imgs_batch = []
                batch_size = len(batch)
                all_imgs = [img for imgs in batch for img in imgs]
                num_negatives = len(all_imgs) // batch_size
                proc_batch = self.processor.process_images(images=all_imgs)
            else:
                raise ValueError(f"Unsupported batch type: {type(batch[0][0])}. Expected str or Image.")
            for k, v in proc_batch.items():
                if isinstance(v, torch.Tensor):                        
                    proc_batch[k] = v.view(batch_size, num_negatives, *v.shape[1:])
                else:
                    proc_batch[k] = v
        else:
            raise ValueError(f"Unsupported batch type: {type(batch[0])}. Expected str or Image.")

        return prefix_keys(proc_batch, key_prefix)