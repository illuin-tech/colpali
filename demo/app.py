import os

import gradio as gr
import torch
from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

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


def search(query: str, ds, images):
    qs = []
    with torch.no_grad():
        batch_query = process_queries(processor, [query], mock_image)
        batch_query = {k: v.to(device) for k, v in batch_query.items()}
        embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # run evaluation
    retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    scores = retriever_evaluator.evaluate(qs, ds)
    best_page = int(scores.argmax(axis=1).item())
    return f"The most relevant page is {best_page}", images[best_page]


def index(file, ds):
    """Example script to run inference with ColPali"""
    images = []
    for f in file:
        images.extend(convert_from_path(f))

    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    return f"Uploaded and converted {len(images)} pages", ds, images


COLORS = ["#4285f4", "#db4437", "#f4b400", "#0f9d58", "#e48ef1"]
# Load model
model_name = "vidore/colpali"
token = os.environ.get("HF_TOKEN")
model = ColPali.from_pretrained(
    "google/paligemma-3b-mix-448", torch_dtype=torch.bfloat16, device_map="cuda", token=token
).eval()
model.load_adapter(model_name)
processor = AutoProcessor.from_pretrained(model_name, token=token)
device = model.device
mock_image = Image.new("RGB", (448, 448), (255, 255, 255))

with gr.Blocks() as demo:
    gr.Markdown("# ColPali: Efficient Document Retrieval with Vision Language Models 📚🔍")
    gr.Markdown("## 1️⃣ Upload PDFs")
    file = gr.File(file_types=["pdf"], file_count="multiple")

    gr.Markdown("## 2️⃣ Convert the PDFs and upload")
    convert_button = gr.Button("🔄 Convert and upload")
    message = gr.Textbox("Files not yet uploaded")
    embeds = gr.State(value=[])
    imgs = gr.State(value=[])

    # Define the actions
    convert_button.click(index, inputs=[file, embeds], outputs=[message, embeds, imgs])

    gr.Markdown("## 3️⃣ Search")
    query = gr.Textbox(placeholder="Enter your query here")
    search_button = gr.Button("🔍 Search")
    message2 = gr.Textbox("Query not yet set")
    output_img = gr.Image()

    search_button.click(search, inputs=[query, embeds, imgs], outputs=[message2, output_img])


if __name__ == "__main__":
    demo.queue(max_size=10).launch(debug=True)
