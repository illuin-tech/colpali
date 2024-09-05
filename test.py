import configue
import importlib
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from colpali_engine.models.late_interaction.colphi3.modeling_phi3_v import Phi3VModel

model = Phi3VModel.from_pretrained("microsoft/Phi-3-vision-128k-instruct", _attn_implementation="eager")


# config_path = Path("scripts/configs/phi/train_colphi_model.yaml")
# config = configue.load(config_path, sub_path="config")
