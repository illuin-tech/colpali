"""Utils for processing images and queries for ColPaLi"""

from typing import List, Optional

from PIL import Image
from transformers import BatchEncoding
from transformers.models.idefics2 import Idefics2Processor


def process_images_idefics(
    processor: Idefics2Processor,
    images: List[Image.Image],
) -> BatchEncoding:
    """
    Process images for ColIdefics2, with an efficient tweak around the Idefics2 processor.
    """
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


def process_queries_idefics(
    processor: Idefics2Processor,
    queries: List[str],
    max_length: int = 50,
    suffix: Optional[str] = None,
) -> BatchEncoding:
    """
    Process queries for ColIdefics2, with an efficient tweak around the Idefics2 processor.
    """
    suffix = suffix or "<end_of_utterance>" * 5
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
