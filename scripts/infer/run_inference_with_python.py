import torch
import typer
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from colpali_engine.utils.image_from_page_utils import load_from_dataset


def main() -> None:
    """Example script to run inference with ColPali"""

    # Load model
    model_name = "vidore/colpali"
    model = ColPali.from_pretrained("google/paligemma-3b-mix-448", torch_dtype=torch.bfloat16, device_map="cuda").eval()
    model.load_adapter(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    # select images -> load_from_pdf(<pdf_path>),  load_from_image_urls(["<url_1>"]), load_from_dataset(<path>)
    images = load_from_dataset("vidore/docvqa_test_subsampled")
    queries = ["From which university does James V. Fiorca come ?", "Who is the japanese prime minister?"]

    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
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
        collate_fn=lambda x: process_queries(processor, x, Image.new("RGB", (448, 448), (255, 255, 255))),
    )

    qs = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # run evaluation
    retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    scores = retriever_evaluator.evaluate(qs, ds)
    print(scores.argmax(axis=1))


if __name__ == "__main__":
    typer.run(main)
