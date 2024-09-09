from __future__ import annotations

from typing import List, cast

from PIL import Image
from transformers import BatchFeature, LlamaTokenizerFast, PaliGemmaProcessor


class ColPali2Processor(PaliGemmaProcessor):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = cast(LlamaTokenizerFast, self.tokenizer)
        self.special_tokens_map = self.tokenizer.special_tokens_map

    def process_text(
        self,
        text: str | List[str],
        padding: str = "longest",
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> BatchFeature:
        """
        Process text inputs for the model.
        If `add_special_tokens` is True (default), the text will be prepended with the <bos> token and
        postpended with " \n".
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
