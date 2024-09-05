"""Utils for processing images and queries for ColPaLi"""

from typing import List, Optional

from PIL import Image
from transformers import BatchFeature
from transformers.models.paligemma import PaliGemmaProcessor


def process_images(
    processor: PaliGemmaProcessor,
    images: List[Image.Image],
    max_length: int = 50,
) -> BatchFeature:
    """
    Process images for ColPaLi, with an efficient tweak around the PaliGemmma processor.
    """
    texts_doc = ["Describe the image."] * len(images)
    images = [image.convert("RGB") for image in images]

    batch_doc = processor(
        text=texts_doc,
        images=images,
        return_tensors="pt",
        padding="longest",
        max_length=max_length + processor.image_seq_length,
    )

    return batch_doc


def process_queries(
    processor: PaliGemmaProcessor,
    queries: List[str],
    max_length: int = 50,
    suffix: Optional[str] = None,
) -> BatchFeature:
    """
    Process queries for ColPaLi, with an efficient tweak around the PaliGemmma processor.
    """
    # NOTE: The image is required for calling PaligemmaProcessor, so we create a mock image here.
    mock_image = Image.new("RGB", (448, 448), (255, 255, 255)).convert("RGB")

    suffix = suffix or "<pad>" * 10
    texts_query: List[str] = []

    for query in queries:
        query = f"Question: {query}"
        query += suffix  # add suffix (pad tokens)
        texts_query.append(query)

    batch_query = processor(
        images=[mock_image] * len(texts_query),
        text=texts_query,
        return_tensors="pt",
        padding="longest",
        max_length=max_length + processor.image_seq_length,
    )
    del batch_query["pixel_values"]

    batch_query["input_ids"] = batch_query["input_ids"][..., processor.image_seq_length :]
    batch_query["attention_mask"] = batch_query["attention_mask"][..., processor.image_seq_length :]

    return batch_query
