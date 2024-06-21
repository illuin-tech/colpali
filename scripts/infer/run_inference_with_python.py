import typer
import torch
from transformers import AutoProcessor

from PIL import Image
import requests

from custom_colbert.models.paligemma_colbert_architecture import ColPali
from custom_colbert.trainer.retrieval_evaluator import CustomEvaluator


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
    # batch_doc = {f"doc_{k}": v for k, v in batch_doc.items()}
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
        max_length=max_length + processor.image_seq_length)
    del batch_query["pixel_values"]

    batch_query["input_ids"] = batch_query["input_ids"][..., processor.image_seq_length:]
    batch_query["attention_mask"] = batch_query["attention_mask"][..., processor.image_seq_length:]

    # batch_query = {f"query_{k}": v for k, v in batch_query.items()}
    return batch_query


def main() -> None:
    """Example script to run inference with ColPali"""

    # Load model
    model_name = "coldoc/colpali-3b-mix-448"
    model = ColPali.from_pretrained("google/paligemma-3b-mix-448", torch_dtype=torch.float16, device_map="cuda").eval()
    model.load_adapter(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    device = model.device

    images = [
        Image.open(requests.get(
            "https://datasets-server.huggingface.co/cached-assets/coldoc/docvqa_test_subsampled/--/fd6f97e6af163270d858f00daf9fa033f805058a/--/default/test/209/image/image.jpg?Expires=1718982078&Signature=HCNcBV3sxtzB2lPDBk4C~GQJQl2qbc6uiDmWRz1JMRoCy~H9KHAU0DriWw6yPtD8LpXqIn11OJ84pNgp4D2uPIZ-Uwg5aFfdDMjtPhEXjoAxKKB~J5DdyDRXriJdSjvcMMnFI1nQ71Bmjd93olgzpfkGn1sDsruIM-BkLv8c9u5KAEJGQd3omvl0i4tJApjkWFGEGp4TTMqAPYT2k1Iv5jJDl5NThRCKHHm0cvQ67zlmD2WSlmPksAcVTLSUf1M3TkIfH9MbnaE1K74a7zRTQ8bT78AwBTVxKznJiyWigq~UtTHLqx~MW3m6x2ZKpFYTezYOJj8wdsU32DcJuIvv~w__&Key-Pair-Id=K3EI6M078Z3AC3",
            stream=True).raw),
        Image.open(requests.get(
            "https://datasets-server.huggingface.co/assets/coldoc/docvqa_test_subsampled/--/fd6f97e6af163270d858f00daf9fa033f805058a/--/default/test/1/image/image.jpg?Expires=1718981760&Signature=V5g~nfy9-ffH3ZlHBUP~-00KEi~DxFQ8fadxX1JykRmjeXbDT-ajvrVsPewwXtG0e~9eYOC0iDuv4DT3TMDv7w-VMz2lrWADAmTX46ic93fidFGOHG4zJymdIUxUCHhxG0if6SNBEDMuiDZFnolH~U8D~33-L3oln~u38YGY0aJJ8q27Tb1cV04O7piXT3JlBPx6Xpy1mMEgulvMY51FWRbwYY8tkZNiucEpqodEnahP1a7ugLM-WwcAh6rL-fjreCBiu955jLEHnnoKSYKT6BoGbZHucDvZ1xdVrZEKCHSCgXiPj5VmRzyY1t9hyORlSEAZF5EEbslpWgifPro0mg__&Key-Pair-Id=K3EI6M078Z3AC3",
            stream=True).raw)
    ]
    queries = ["When is the ASMBR meeting ?", "What is the capital of Italy?"]

    # run inference - queries
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        queries,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda x: process_queries(processor, x, images[0]),
    )

    qs = []
    for batch_query in dataloader:
        with torch.no_grad():
            # put on device
            batch_query = {k: v.to(device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )
    ds = []
    for batch_doc in dataloader:
        with torch.no_grad():
            # put on device
            batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    # run evaluation
    retriever_evaluator = CustomEvaluator(is_multi_vector=True)

    scores = retriever_evaluator.evaluate(qs, ds)
    print(scores.shape)
    print(scores)


if __name__ == "__main__":
    typer.run(main)
