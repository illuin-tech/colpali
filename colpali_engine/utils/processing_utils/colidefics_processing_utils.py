"""Utils for processing images and queries for ColPaLi"""
from typing import List

from PIL import Image
from transformers import BatchFeature


def process_images(processor, images: List[Image]) -> BatchFeature:
    texts_doc = []
    images = [image.convert("RGB") for image in images]

    for _ in images:
        messages_doc = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image."},
                    {"type": "image"},
                ],
            },
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


def process_queries(
    processor, queries: List[str], max_length: int = 50, suffix: str = "default_suffix"
) -> BatchFeature:
    if suffix == "default_suffix":
        suffix = "<end_of_utterance>" * 5
    texts_query = []
    for query in queries:
        messages_query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Question: {query}" + suffix,
                    },
                ],
            },
        ]
        text_query = processor.apply_chat_template(messages_query, add_generation_prompt=False).strip()
        texts_query.append(text_query)

    batch_query = processor(
        text=texts_query,
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
    )
    return batch_query
