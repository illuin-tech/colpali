from typing import Annotated, cast

import torch
import typer
from transformers.models.idefics3 import Idefics3ForConditionalGeneration

from colpali_engine.models.idefics3.colidefics3.modeling_colidefics3 import ColIdefics3
from colpali_engine.utils.torch_utils import get_torch_device


def main(
    vlm_backbone_name: Annotated[str, typer.Option(help="The name of the VLM backbone model to use.")],
    new_base_model_name: Annotated[str, typer.Option(help="The name of the base model to push to the hub.")],
):
    """
    Publish the base ColIdefics3 model to the hub.

    Args:
    - vlm_backbone_name (str): The name of the VLM backbone model to use.
    - new_base_model_name (str): The name of the base model to push to the hub.

    Example usage:
    ```bash
    python colpali_engine/models/idefics3/colidefics3/publish_base_colidefics3.py \
        --vlm-backbone-name smol-explorers/SmolVLM-256M-Base-25750 \
        --new-base-model-name vidore/colsmolvlm-256M-base
    ```
    """
    device = get_torch_device("auto")

    vlm_backbone = cast(
        Idefics3ForConditionalGeneration,
        Idefics3ForConditionalGeneration.from_pretrained(
            vlm_backbone_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ),
    ).eval()

    model = ColIdefics3(config=vlm_backbone.config).to(device).to(torch.bfloat16).eval()

    # Copy pre-trained weights from old model
    model.load_state_dict(vlm_backbone.state_dict(), strict=False)

    model.push_to_hub(new_base_model_name, private=True)

    return


if __name__ == "__main__":
    typer.run(main)
