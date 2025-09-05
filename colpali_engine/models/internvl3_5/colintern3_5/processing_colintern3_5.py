from typing import ClassVar, List, Optional, Tuple, Union
import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature
from transformers.models.internvl import InternVLProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

class ColIntern3_5Processor(BaseVisualRetrieverProcessor, InternVLProcessor):  # noqa: N801
    """
    Processor for the ColIntern3.5 model.
    Combines an image processor and tokenizer to prepare inputs for ColIntern3.5, following ColPali conventions.
    
    Args:
        *args: Arguments for the InternVLProcessor.
        **kwargs: Keyword arguments for the InternVLProcessor (e.g., tokenizer or processor initialization parameters).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure text tokenizer pads on the right
        self.tokenizer.padding_side = "right"
        
        # Set image sequence length (number of tokens per patch)
        self.image_seq_length = getattr(self.tokenizer, 'image_seq_length', 256)
        
        # Set up image token attributes if they exist
        self.image_token = getattr(self.tokenizer, 'context_image_token', '<IMG_CONTEXT>')
        self.start_image_token = getattr(self.tokenizer, 'start_image_token', '')
        self.end_image_token = getattr(self.tokenizer, 'end_image_token', '')
        
        # Store the special image token ID for quick access
        if hasattr(self.tokenizer, "context_image_token_id"):
            self.image_token_id = self.tokenizer.context_image_token_id
        elif hasattr(self.tokenizer, "image_token_id"):
            self.image_token_id = self.tokenizer.image_token_id

    @property
    def query_augmentation_token(self) -> str:
        """
        Return the query augmentation token.
        Query augmentation buffers are used as reasoning buffers during inference.
        """
        return self.tokenizer.pad_token

    @classmethod
    def from_pretrained(cls, *args, device_map: Optional[str] = None, **kwargs):
        instance = super().from_pretrained(*args, device_map=device_map, **kwargs)
        # Optionally limit visual tokens by adjusting image processor's configuration (if provided)
        if "max_num_visual_tokens" in kwargs:
            max_tokens = kwargs["max_num_visual_tokens"]
            # For GotOCR2 image processor, we need to limit the max_patches more aggressively
            if hasattr(instance.image_processor, "max_patches"):
                # Reduce max_patches to a much lower value to actually limit visual tokens
                target_patches = min(max_tokens // 256, 10)  # Much more conservative estimate
                instance.image_processor.max_patches = max(1, target_patches)
                # Also limit the image size to help reduce tokens
                if hasattr(instance.image_processor, "size"):
                    # Reduce image size for token limiting
                    instance.image_processor.size = {"height": 224, "width": 224}
        return instance

    def process_images(self, images: List[Image.Image]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process a batch of images for input to ColIntern3.5.
        Returns a BatchFeature with keys including `input_ids`, `attention_mask`, and `pixel_values`.
        """
        images = [img.convert("RGB") for img in images]
        
        # Use the standard InternVL processor approach by calling self() directly
        # This will handle image tokens correctly using _insert_media_placeholders
        placeholders = [self.image_token for _ in range(len(images))]
        
        # Call the parent processor which handles the token replacement properly
        batch = self(text=placeholders, images=images, padding="longest", return_tensors="pt")
        
        # Convert pixel_values to bfloat16 to match model dtype
        if "pixel_values" in batch and isinstance(batch["pixel_values"], torch.Tensor):
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        elif "pixel_values" in batch and isinstance(batch["pixel_values"], list):
            batch["pixel_values"] = [p.to(torch.bfloat16) for p in batch["pixel_values"]]
        
        # If pixel_values are a list (images have different sizes), pad them to uniform spatial dimensions
        pixel_values = batch["pixel_values"]
        if isinstance(pixel_values, list):
            max_h = max(p.shape[-2] for p in pixel_values)
            max_w = max(p.shape[-1] for p in pixel_values)
            padded_tensors = []
            for p in pixel_values:
                c, h, w = p.shape
                padded = torch.zeros((c, max_h, max_w), dtype=p.dtype)
                padded[:, :h, :w] = p
                padded_tensors.append(padded)
            batch["pixel_values"] = torch.stack(padded_tensors, dim=0)
        return batch

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process a batch of text queries for input to ColIntern3.5.
        """
        return self(text=texts, return_tensors="pt", padding="longest")

    # Alias for process_texts
    def process_queries(self, queries: List[str]) -> Union[BatchFeature, BatchEncoding]:
        return self.process_texts(queries)

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

    def score_multi_vector(self, qs: Union[List[torch.Tensor], torch.Tensor], ps: Union[List[torch.Tensor], torch.Tensor], device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """
        Compute ColBERT-style MaxSim score between multi-vector queries and passages.
        Each query embedding and passage embedding can be a list of token embeddings (or a tensor of shape [seq_length, dim]).
        Returns a tensor of scores with shape (len(qs), len(ps)).
        """
        # Use the base class implementation which returns the correct shape
        return super().score_multi_vector(qs, ps, device=device)

    def get_n_patches(self, image_size: Tuple[int, int], spatial_merge_size: int) -> Tuple[int, int]:
        """
        Compute the number of patch tokens (n_patches_x, n_patches_y) for an image of given (height, width) in pixels.
        """
        patch_size = getattr(self.image_processor, "patch_size", 14)
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        height, width = image_size
        factor = patch_size * spatial_merge_size
        # Scale down large images to max 448 on the longer side to limit patch count
        max_side = 448
        if max(height, width) > max_side:
            scale = max_side / max(height, width)
            height = int(height * scale)
            width = int(width * scale)
        # Align to the nearest lower multiple of factor (minimum one factor)
        height_new = max(factor, (height // factor) * factor)
        width_new = max(factor, (width // factor) * factor)
        n_patches_y = height_new // patch_size // spatial_merge_size
        n_patches_x = width_new // patch_size // spatial_merge_size
        return (n_patches_x, n_patches_y)

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """
        Generate a mask indicating which positions in the input IDs correspond to image patch tokens.
        """
        return batch_images["input_ids"] == getattr(self, "image_token_id", self.tokenizer.context_image_token_id)
