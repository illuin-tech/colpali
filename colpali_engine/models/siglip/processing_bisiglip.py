from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature
from transformers.models.siglip import SiglipProcessor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class BiSiglipProcessor(BaseVisualRetrieverProcessor, SiglipProcessor):  # noqa: N801
    """
    Processor for BiSiglip
    """

    query_augmentation_token: ClassVar[str] = "<pad>"

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
        """
        return self(
            text=texts,
            return_tensors="pt",
            padding="max_length",  # the model was trained with max_length padding
            max_length=64,
            truncation=True,
        )

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Args:
            images: List of PIL images.
        """
        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            images=images,
            return_tensors="pt",
            padding="longest",  # the model was trained with max_length padding
        )
        return batch_doc

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the cosine similarity for the given query and passage embeddings.
        """
        return self.score_single_vector(qs, ps, device=device)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        The `spatial_merge_size` is the number of patches that will be merged spatially. It is stored in
        as a `Qwen2VLForConditionalGeneration` attribute under `model.spatial_merge_size`.
        """
        raise NotImplementedError("BiSiglip does not support the `get_n_patches` method. ")