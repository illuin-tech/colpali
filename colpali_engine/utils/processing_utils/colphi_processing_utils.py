"""Utils for processing images and queries for ColPhi"""

from typing import List, Optional

from PIL import Image
from transformers import BatchFeature


def process_images(processor, images: List[Image]) -> BatchFeature:
    texts_doc = []
    placeholder = ""

    for _ in images:
        messages_doc = [
            {"role": "user", "content": "<|image_1|>\nDescribe the image."},
        ]
        text_doc = processor.apply_chat_template(messages_doc, add_generation_prompt=False)
        texts_doc.append(text_doc.strip())

    batch_doc = processor(
        text=texts_doc,
        images=images,
        return_tensors="pt",
        padding="longest",
    )
    return batch_doc


def process_queries(processor, queries: List[str], max_length: int = 50, suffix: Optional[str] = None) -> BatchFeature:
    suffix = suffix or "<|endoftext|>" * 5
    texts_query = []
    for query in queries:
        messages_query = [
            {"role": "user", "content": f"Question: {query}" + suffix},
        ]
        text_query = processor.tokenizer.apply_chat_template(
            messages_query, tokenize=False, add_generation_prompt=False
        )
        print(text_query)
        texts_query.append(text_query)

    batch_query = processor(
        text=texts_query,
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
    )
    return batch_query
