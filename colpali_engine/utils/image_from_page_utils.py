import requests
from PIL import Image


def load_from_pdf(pdf_path: str):
    from pdf2image import convert_from_path

    images = convert_from_path(pdf_path)
    return images


def load_from_image_urls(urls: str):
    images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
    return images


def load_from_dataset(dataset):
    from datasets import load_dataset

    dataset = load_dataset(dataset, split="test")
    return dataset["image"]
