from typing import ClassVar, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, Idefics3Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColIdefics3Processor(BaseVisualRetrieverProcessor, Idefics3Processor):
    """
    Processor for ColIdefics3.
    """

    query_prefix: ClassVar[str] = "Query: "
    query_augmentation_token: ClassVar[str] = "<end_of_utterance>"
    image_token: ClassVar[str] = "<image>"
    visual_prompt_prefix: ClassVar[str] = "<|im_start|>user\n<image>Describe the image.<end_of_utterance>"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    @staticmethod
    def reduce_tensor_type(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.int64:
                    tensor_dict[key] = value.to(torch.int32)
                elif value.dtype == torch.float64:
                    tensor_dict[key] = value.to(torch.float32)
        return tensor_dict

    def process_images(
        self,
        images: List[Image.Image],
        context_prompts: Optional[List[str]] = None,
    ) -> BatchEncoding:
        """
        Process images for ColIdefics3.

        Args:
            images: List of PIL images.
            context_prompts: List of optional context prompts, i.e. some text description of the context of the image.
        """

        texts_doc: List[str] = []
        images = [[image.convert("RGB")] for image in images]

        if context_prompts:
            if len(images) != len(context_prompts):
                raise ValueError("Length of images and context prompts must match.")
            texts_doc = context_prompts
        else:
            texts_doc = [self.visual_prompt_prefix] * len(images)

        batch_doc = self(
            text=texts_doc,
            images=images,
            return_tensors="pt",
            padding="longest",
        )
        batch_doc = self.reduce_tensor_type(batch_doc)
        return batch_doc

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchEncoding:
        """
        Process queries for ColIdefics3.
        """
        if suffix is None:
            suffix = self.query_augmentation_token * 10
        texts_query: List[str] = []

        for query in queries:
            query = self.query_prefix + query + suffix + "\n"
            texts_query.append(query)

        batch_query = self.tokenizer(
            text=texts_query,
            return_tensors="pt",
            padding="longest",
        )
        batch_query = self.reduce_tensor_type(batch_query)
        return batch_query

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
    ) -> Tuple[int, int]:
        raise NotImplementedError("This method is not implemented for ColIdefics3.")
