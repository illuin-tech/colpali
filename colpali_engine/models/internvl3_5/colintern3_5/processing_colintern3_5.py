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
        # Store the special image token ID for quick access
        if hasattr(self.tokenizer, "context_image_token_id"):
            self.image_token_id = self.tokenizer.context_image_token_id

    @classmethod
    def from_pretrained(cls, *args, device_map: Optional[str] = None, **kwargs):
        instance = super().from_pretrained(*args, device_map=device_map, **kwargs)
        # Optionally limit visual tokens by adjusting image processor's max pixels (if provided)
        if "max_num_visual_tokens" in kwargs:
            max_tokens = kwargs["max_num_visual_tokens"]
            # Each merged image patch covers 28x28 pixels (for patch_size 14 and merge size 2)
            instance.image_processor.max_pixels = max_tokens * 28 * 28
            if hasattr(instance.image_processor, "size") and isinstance(instance.image_processor.size, dict):
                instance.image_processor.size["longest_edge"] = instance.image_processor.max_pixels
        return instance

    def process_images(self, images: List[Image.Image]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process a batch of images for input to ColIntern3.5.
        Returns a BatchFeature with keys including `input_ids`, `attention_mask`, and `pixel_values`.
        """
        images = [img.convert("RGB") for img in images]
        # Prepare text placeholders for images using the special image token
        placeholder = getattr(self, "image_token", self.tokenizer.context_image_token)
        batch = self(text=[placeholder] * len(images), images=images, padding="longest", return_tensors="pt")
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

    def score_multi_vector(self, qs: Union[List[torch.Tensor], torch.Tensor], ps: Union[List[torch.Tensor], torch.Tensor], device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """
        Compute ColBERT-style MaxSim score between multi-vector queries and passages.
        Each query embedding and passage embedding can be a list of token embeddings (or a tensor of shape [seq_length, dim]).
        Returns a tensor of scores (one per query-passage pair).
        """
        # Normalize inputs to lists of tensors
        if isinstance(qs, torch.Tensor):
            qs = [emb for emb in qs]
        if isinstance(ps, torch.Tensor):
            ps = [emb for emb in ps]
        # Move embeddings to the target device if specified
        if device is not None:
            device_obj = torch.device(device)
            qs = [emb.to(device_obj) for emb in qs]
            ps = [emb.to(device_obj) for emb in ps]
        scores: List[float] = []
        if len(qs) == len(ps):
            # Pairwise scoring for each query-passage pair
            for q_emb, p_emb in zip(qs, ps):
                sim = torch.matmul(q_emb, p_emb.T)  # (len_q_tokens, len_p_tokens)
                max_sim, _ = sim.max(dim=1)         # max similarity for each query token
                scores.append(max_sim.sum().item())
        elif len(qs) == 1 and len(ps) > 1:
            # One query vs many passages
            q_emb = qs[0]
            for p_emb in ps:
                sim = torch.matmul(q_emb, p_emb.T)
                max_sim, _ = sim.max(dim=1)
                scores.append(max_sim.sum().item())
        elif len(ps) == 1 and len(qs) > 1:
            # Many queries vs one passage
            p_emb = ps[0]
            for q_emb in qs:
                sim = torch.matmul(q_emb, p_emb.T)
                max_sim, _ = sim.max(dim=1)
                scores.append(max_sim.sum().item())
        else:
            raise ValueError("Inputs for queries and passages must have compatible batch sizes.")
        return torch.tensor(scores, dtype=torch.float32, device=(device_obj if device is not None else None))

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
