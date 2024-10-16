from typing import cast

import torch
import typer

from colpali_engine.models.paligemma.colpali.modeling_colpali import ColPali
from colpali_engine.models.paligemma.colpali_duo.configuration_colpali_duo import ColPaliDuoConfig
from colpali_engine.models.paligemma.colpali_duo.modeling_colpali_duo import ColPaliDuo
from colpali_engine.utils.torch_utils import get_torch_device


def main():
    """
    Publish the base ColPaliDuo model to the hub.
    """
    base_colpali_duo_name = "vidore/colpali-duo-base-0.1"
    device = get_torch_device("auto")

    old_model = cast(
        ColPali,
        ColPali.from_pretrained(
            "vidore/colpali-v1.2",
            torch_dtype=torch.bfloat16,
            device_map=device,  # or "mps" if on Apple Silicon
        ),
    ).eval()

    model_config = ColPaliDuoConfig(
        **old_model.config.to_dict(),
        single_vector_projector_dim=1024,
        single_vector_pool_strategy="mean",
        multi_vector_projector_dim=128,
    )

    model = ColPaliDuo(config=model_config).to(device).to(torch.bfloat16).eval()

    model.push_to_hub(base_colpali_duo_name, private=True)

    return


if __name__ == "__main__":
    typer.run(main)
