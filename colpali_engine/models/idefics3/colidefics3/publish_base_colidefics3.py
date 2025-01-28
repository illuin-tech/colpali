from typing import Annotated

import torch
import typer
from transformers import AutoProcessor

from colpali_engine.models.idefics3.colidefics3.modeling_colidefics3 import ColIdefics3
from colpali_engine.utils.torch_utils import get_torch_device


def main(
    model_name: Annotated[
        str, typer.Option(help="The name of the VLM backbone model to use.")
    ],
    new_base_model_name: Annotated[
        str, typer.Option(help="The name of the base model to push to the hub.")
    ],
):
    """
    Publish the base ColIdefics3 model to the hub.

    Args:
    - model_name (str): The name of the VLM backbone model to use.
    - new_base_model_name (str): The name of the base model to push to the hub.

    Example usage:
    ```bash
    python colpali_engine/models/idefics3/colidefics3/publish_base_colidefics3.py \
        --model-name vidore/SmolVLM-Instruct-500M \
        --new-base-model-name vidore/MyNew_ColSmolVLM-Instruct-500M
    ```
    """
    device = get_torch_device("auto")

    model = ColIdefics3.from_pretrained(model_name).to(device).to(torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    model.push_to_hub(new_base_model_name, private=True)
    processor.push_to_hub(new_base_model_name, private=True)

    return


if __name__ == "__main__":
    typer.run(main)
