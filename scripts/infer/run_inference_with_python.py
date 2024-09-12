from typing import cast

import torch
import typer
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import get_torch_device


def main() -> None:
    """
    Example script to run inference with ColPali
    """

    device = get_torch_device("auto")
    print(f"Device used: {device}")

    # Define adapter name
    base_model_name = "vidore/colpaligemma-3b-pt-448-base"
    adapter_name = "vidore/colpali-v1.2"

    # Load model
    model = ColPali.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    model.load_adapter(adapter_name)
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained("google/paligemma-3b-mix-448"))

    if not isinstance(processor, BaseVisualRetrieverProcessor):
        raise ValueError("Processor should be a BaseVisualRetrieverProcessor")

    images = cast(Dataset, load_dataset("vidore/docvqa_test_subsampled", split="test"))["image"]
    queries = ["From which university does James V. Fiorca come ?", "Who is the japanese prime minister?"]

    # Run inference - docs
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

    # Run inference - queries
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
