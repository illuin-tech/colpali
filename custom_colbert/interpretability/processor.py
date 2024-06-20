from __future__ import annotations

from dataclasses import dataclass
from typing import List, cast

import torch
from PIL import Image
from transformers import LlamaTokenizerFast, PaliGemmaProcessor


@dataclass
class ColPaliTextInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: torch.device) -> ColPaliTextInput:
        return ColPaliTextInput(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
        )


@dataclass
class ColPaliImageInput:
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: str | torch.device) -> ColPaliImageInput:
        return ColPaliImageInput(
            input_ids=self.input_ids.to(device),
            pixel_values=self.pixel_values.to(device),
            attention_mask=self.attention_mask.to(device),
        )


class ColPaliProcessor:
    def __init__(self, processor: PaliGemmaProcessor):
        self.processor = processor
        self.tokenizer = cast(LlamaTokenizerFast, self.processor.tokenizer)  # type: ignore

    @staticmethod
    def from_pretrained(model_name: str) -> ColPaliProcessor:
        return ColPaliProcessor(processor=cast(PaliGemmaProcessor, PaliGemmaProcessor.from_pretrained(model_name)))

    def process_text(
        self,
        text: str | List[str],
        padding: str = "longest",
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> ColPaliTextInput:
        """
        Process text inputs for the model.
        If `add_special_tokens` is True (default), the text will be prepended with the <bos> token and appended with " \n".
        """
        if add_special_tokens:
            if isinstance(text, str):
                text = self.tokenizer.bos_token + text + "\n"
            elif isinstance(text, list):
                text = [self.tokenizer.bos_token + t + "\n" for t in text]
            else:
                raise ValueError("text must be a string or a list of strings.")

        batch_output = self.tokenizer(
            text, padding=padding, return_tensors=return_tensors, add_special_tokens=add_special_tokens
        )

        return ColPaliTextInput(
            input_ids=cast(torch.Tensor, batch_output["input_ids"]),
            attention_mask=cast(torch.Tensor, batch_output["attention_mask"]),
        )

    def process_image(
        self,
        image: Image.Image | List[Image.Image],
        padding: str = "longest",
        do_convert_rgb: bool = True,
        return_tensors: str = "pt",
        add_special_prompt: bool = True,
    ) -> ColPaliImageInput:
        # NOTE: The special prompt was used at training time,
        special_prompt = "Describe the image." if add_special_prompt else None
        if isinstance(image, Image.Image):
            text_input = [special_prompt]
        elif isinstance(image, list):
            text_input = [special_prompt] * len(image)
        else:
            raise ValueError("image must be a PIL Image or a list of PIL Images.")

        batch_output = self.processor(
            text=text_input,
            images=image,
            padding=padding,
            do_convert_rgb=do_convert_rgb,
            return_tensors=return_tensors,
        )

        if add_special_prompt:
            return ColPaliImageInput(
                input_ids=batch_output["input_ids"],
                pixel_values=batch_output["pixel_values"],
                attention_mask=batch_output["attention_mask"],
            )
        else:
            return ColPaliImageInput(
                input_ids=batch_output["input_ids"][:, : self.processor.image_seq_length],
                pixel_values=batch_output["pixel_values"][:, : self.processor.image_seq_length],
                attention_mask=batch_output["attention_mask"][:, : self.processor.image_seq_length],
            )

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
