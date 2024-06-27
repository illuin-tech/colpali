# Utils for processing images and queries for ColPaLi


def process_images(processor, images, max_length: int = 50):
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


def process_queries(processor, queries, mock_image, max_length: int = 50):
    texts_query = []
    for query in queries:
        query = f"Question: {query}<unused0><unused0><unused0><unused0><unused0>"
        texts_query.append(query)

    batch_query = processor(
        images=[mock_image.convert("RGB")] * len(texts_query),
        # NOTE: the image is not used in batch_query but it is required for calling the processor
        text=texts_query,
        return_tensors="pt",
        padding="longest",
        max_length=max_length + processor.image_seq_length,
    )
    del batch_query["pixel_values"]

    batch_query["input_ids"] = batch_query["input_ids"][..., processor.image_seq_length :]
    batch_query["attention_mask"] = batch_query["attention_mask"][..., processor.image_seq_length :]
    return batch_query
