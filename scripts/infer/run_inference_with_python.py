from typing import cast

import torch
import typer
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from colpali_engine.models import ColPali
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


def main() -> None:
    """Example script to run inference with ColPali"""

    # Load model
    model_name = "vidore/colpali-v1.2"
    model = ColPali.from_pretrained(
        "vidore/colpaligemma-3b-pt-448-base", torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    model.load_adapter(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    if not isinstance(processor, BaseVisualRetrieverProcessor):
        raise ValueError("Processor should be a BaseVisualRetrieverProcessor")

    images = cast(Dataset, load_dataset("vidore/docvqa_test_subsampled", split="test"))["image"]
    queries = ["From which university does James V. Fiorca come ?", "Who is the japanese prime minister?"]

    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    ds = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    # run inference - queries
    dataloader = DataLoader(
        queries,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )

    qs = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # run evaluation
    scores = processor.score(qs, ds)
    print(scores.argmax(axis=1))


if __name__ == "__main__":
    typer.run(main)
