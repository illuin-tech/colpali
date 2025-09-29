from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Idefics3Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColEuroVBertProcessor(BaseVisualRetrieverProcessor, Idefics3Processor):
    """
    Processor for ColIdefics3.
    """

    query_augmentation_token: ClassVar[str] = "<|end_of_text|>"
    image_token: ClassVar[str] = "<image>"
    visual_prompt_prefix: ClassVar[str] = "<|begin_of_text|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"

    def __init__(self, *args, image_seq_len=64, **kwargs):
        super().__init__(*args, image_seq_len=image_seq_len, **kwargs)
        self.tokenizer.padding_side = "left"

    # @property
    # def image_token_id(self) -> int:
    #     return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def process_images(
        self,
        images: List[Image.Image],
        contexts: Optional[List[str]] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for ColVBert.

        Args:
            images: List of PIL images.
            contexts: List of optional context prompts, i.e. some text description of the context of the image.
        """
        # if contexts is None:
        #     contexts = [self.visual_prompt_prefix] * len(images)
        contexts = [self.visual_prompt_prefix] * len(images)

        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=contexts,
            images=images,
            padding="longest",
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        )
        return batch_doc

    def process_texts(
        self,
        texts: List[str],
        max_length: int = 50,
        contexts: Optional[List[str]] = None,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for ColVBert.

        NOTE: `max_length` is not used and kept only for trainer compatibility.
        """
        if suffix is None:
            suffix = self.query_augmentation_token * 10
        if contexts is None:
            contexts = [""] * len(texts)

        prompts = [context + text + suffix for context, text in zip(contexts, texts)]

        batch_texts = self(
            text=prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=4096,
        )

        return batch_texts

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
