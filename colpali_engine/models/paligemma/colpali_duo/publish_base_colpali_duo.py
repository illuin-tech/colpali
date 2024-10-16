from typing import cast

import torch
import typer
from transformers.models.paligemma import PaliGemmaForConditionalGeneration

from colpali_engine.models.paligemma.colpali_duo.configuration_colpali_duo import ColPaliDuoConfig
from colpali_engine.models.paligemma.colpali_duo.modeling_colpali_duo import ColPaliDuo
from colpali_engine.utils.torch_utils import get_torch_device


def main():
    """
    Publish the base ColPaliDuo model to the hub.
    """
    base_colpali_duo_name = "vidore/colpali-duo-base"
    device = get_torch_device("auto")

    # Load old model
    old_model = cast(
        PaliGemmaForConditionalGeneration,
        PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-mix-448",
            torch_dtype=torch.bfloat16,
            device_map=device,
        ),
    ).eval()

    # Load new model
    model_config = ColPaliDuoConfig(
        **old_model.config.to_dict(),
        single_vector_projector_dim=1024,
        single_vector_pool_strategy="mean",
        multi_vector_projector_dim=128,
    )
    model = ColPaliDuo(config=model_config).to(device).to(old_model.dtype).eval()

    # Copy pre-trained weights from old model
    model.load_state_dict(old_model.state_dict(), strict=False)

    # Push to hub
    model.push_to_hub(base_colpali_duo_name, private=True)

    return


if __name__ == "__main__":
    typer.run(main)
