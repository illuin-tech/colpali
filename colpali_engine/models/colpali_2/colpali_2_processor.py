from __future__ import annotations

from typing import List, cast

import torch
from PIL import Image
from transformers import BatchFeature, LlamaTokenizerFast, PaliGemmaProcessor


class ColPali2Processor(PaliGemmaProcessor):
    def __init__(
        self,
        *args,
        cls_token: str = "<unused1>",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = cast(LlamaTokenizerFast, self.tokenizer)  # type: ignore
        self.special_tokens_map = self.tokenizer.special_tokens_map
        self.cls_token = cls_token
        if self.cls_token not in self.tokenizer.added_tokens_decoder:
            raise ValueError(f"The tokenizer should have an `{cls_token}` token to be used as the <cls> token.")
        self.special_tokens_map["cls_token"] = self.cls_token
        self.cls_token_id = cast(int, self.tokenizer.convert_tokens_to_ids(self.cls_token))

    def process_text(
        self,
        text: str | List[str],
        padding: str = "longest",
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> BatchFeature:
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

        tokenized_outputs = self.tokenizer(
            text, padding=padding, return_tensors=return_tensors, add_special_tokens=add_special_tokens
        )

        return BatchFeature(
            data={
                "input_ids": tokenized_outputs["input_ids"],
                "attention_mask": tokenized_outputs["attention_mask"],
            }
        )

    def process_image(
        self,
        image: Image.Image | List[Image.Image],
        padding: str = "longest",
        do_convert_rgb: bool = True,
        return_tensors: str = "pt",
        add_instruction_prompt: bool = True,
    ) -> BatchFeature:
        # NOTE: The special prompt was used at training time. If used, it will be appended at the end of the input_ids.
        special_prompt = "Describe the image." if add_instruction_prompt else None

        if isinstance(image, Image.Image):
            text_input = [special_prompt]
        elif isinstance(image, list):
            text_input = [special_prompt] * len(image)
        else:
            raise ValueError("image must be a PIL Image or a list of PIL Images.")

        batch_output = self(
            text=text_input,  # type: ignore
            images=image,
            padding=padding,
            do_convert_rgb=do_convert_rgb,
            return_tensors=return_tensors,
        )

        batch_output["input_ids"] = batch_output["input_ids"][:, : self.image_seq_length]
        batch_output["pixel_values"] = batch_output["pixel_values"][:, : self.image_seq_length]
        batch_output["attention_mask"] = batch_output["attention_mask"][:, : self.image_seq_length]

        return batch_output

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def is_cls_token_first(self, input_ids: torch.Tensor) -> bool:
        """
        Check if the first token in each sequence of the batch is the CLS token.

        Inputs:
        - input_ids (torch.Tensor): The input_ids tensor (batch_size, sequence_length).
        """
        if input_ids.dim() != 2:
            raise ValueError("`input_ids` must be a 2D tensor.")
        return cast(bool, torch.all(input_ids[:, 0] == self.cls_token_id).item())
