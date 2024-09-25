import pprint
from typing import List, cast

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device


def main():
    """
    Example script to run inference with ColPali.
    """

    device = get_torch_device("auto")
    print(f"Device used: {device}")

    # Model name
    model_name = "vidore/colpali-v1.2"

    # Load model
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    # Load processor
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))

    if not isinstance(processor, BaseVisualRetrieverProcessor):
        raise ValueError("Processor should be a BaseVisualRetrieverProcessor")

    # NOTE: Only the first 16 images are used for demonstration purposes
    dataset = cast(Dataset, load_dataset("vidore/docvqa_test_subsampled", split="test[:16]"))
    images = dataset["image"]

    # Select a few queries for demonstration purposes
    query_indices = [12, 15]
    queries = [dataset[idx]["query"] for idx in query_indices]
    print("Selected queries:")
    pprint.pprint(dict(zip(query_indices, queries)))

    # Run inference - docs
    dataloader = DataLoader(
        dataset=ListDataset[str](images),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    ds: List[torch.Tensor] = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    # Run inference - queries
    dataloader = DataLoader(
        dataset=ListDataset[str](queries),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )

    qs: List[torch.Tensor] = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # Run scoring
    scores = processor.score(qs, ds).cpu().numpy()
    idx_top_1 = scores.argmax(axis=1)
    print("Indices of the top-1 retrieved documents for each query:", idx_top_1)

    # Sanity check
    if idx_top_1.tolist() == query_indices:
        print("The top-1 retrieved documents are correct.")
    else:
        print("The top-1 retrieved documents are incorrect.")

    return


if __name__ == "__main__":
    main()
