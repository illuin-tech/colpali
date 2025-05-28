import asyncio
import base64
import os
from io import BytesIO
from typing import Any, List

import aiohttp
import torch
from PIL import Image
from tqdm.asyncio import tqdm_asyncio


class IlluinAPIModelWrapper:
    def __init__(
        self,
        model_name: str,
        **kwargs: Any,
    ):
        """Wrapper for Illuin API embedding model"""
        self.model_name = model_name
        self.url = model_name
        self.HEADERS = {
            "Accept": "application/json",
            "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def convert_image_to_base64(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def post_images(self, session: aiohttp.ClientSession, encoded_images: List[str]):
        payload = {"inputs": {"images": encoded_images}}
        async with session.post(self.url, headers=self.HEADERS, json=payload) as response:
            return await response.json()

    async def post_queries(self, session: aiohttp.ClientSession, queries: List[str]):
        payload = {"inputs": {"queries": queries}}
        async with session.post(self.url, headers=self.HEADERS, json=payload) as response:
            return await response.json()

    async def call_api_queries(self, queries: List[str]):
        embeddings = []
        semaphore = asyncio.Semaphore(16)
        async with aiohttp.ClientSession() as session:

            async def sem_post(batch):
                async with semaphore:
                    return await self.post_queries(session, batch)

            tasks = [asyncio.create_task(sem_post([batch])) for batch in queries]

            # ORDER-PRESERVING
            results = await tqdm_asyncio.gather(*tasks, desc="Query batches")

            for result in results:
                embeddings.extend(result.get("embeddings", []))

        return embeddings

    async def call_api_images(self, images_b64: List[str]):
        embeddings = []
        semaphore = asyncio.Semaphore(16)

        async with aiohttp.ClientSession() as session:

            async def sem_post(batch):
                async with semaphore:
                    return await self.post_images(session, batch)

            tasks = [asyncio.create_task(sem_post([batch])) for batch in images_b64]

            # ORDER-PRESERVING
            results = await tqdm_asyncio.gather(*tasks, desc="Doc batches")

            for result in results:
                embeddings.extend(result.get("embeddings", []))

        return embeddings

    def forward_queries(self, queries: List[str]) -> torch.Tensor:
        response = asyncio.run(self.call_api_queries(queries))
        return response

    def forward_passages(self, passages: List[Image.Image]) -> torch.Tensor:
        response = asyncio.run(self.call_api_images([self.convert_image_to_base64(doc) for doc in passages]))
        return response


if __name__ == "__main__":
    # Example usage

    client = IlluinAPIModelWrapper(
        model_name="https://sxeg6spz1yy8unh7.us-east-1.aws.endpoints.huggingface.cloud",
    )

    embed_queries = client.forward_queries(["What is the capital of France?", "Explain quantum computing."])

    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (128, 128), color="black"),
    ]

    embed_images = client.forward_passages(images)

    print("Query embeddings shape:", len(embed_queries))
    print("Image embeddings shape:", len(embed_images))
