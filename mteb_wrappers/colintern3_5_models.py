"""
MTEB wrapper for ColIntern3.5 models.
"""

from __future__ import annotations

import logging
from functools import partial

import torch
from transformers.utils.import_utils import is_flash_attn_2_available

from mteb.model_meta import ModelMeta
from mteb.models.colpali_models import COLPALI_TRAINING_DATA, ColPaliEngineWrapper
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class ColIntern3_5Wrapper(ColPaliEngineWrapper):
    """Wrapper for ColIntern3.5 model."""

    def __init__(
        self,
        model_name: str = "experiments/colintern3_5-1B-lora",
        revision: str | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,  # Change default to bfloat16
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColIntern3_5, ColIntern3_5Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model from the specified path
        # For PEFT checkpoints, we need to load the base model first and then the adapter
        if model_name.startswith("experiments/"):
            # This is a local PEFT checkpoint
            base_model_name = "OpenGVLab/InternVL3_5-1B-HF"
            
            # First load the base model without PEFT
            self.mdl = ColIntern3_5.from_pretrained(
                base_model_name,
                device_map=self.device,
                torch_dtype=torch_dtype,
                **kwargs,
            )
            
            # Then load the PEFT adapter using the proper approach
            from peft import PeftModel
            self.mdl = PeftModel.from_pretrained(
                self.mdl, 
                model_name,
                torch_dtype=torch_dtype,
                is_trainable=False  # Set to False for inference
            )
            
            # Merge the adapter weights into the base model for inference
            # This ensures the custom_text_proj gets the trained weights
            self.mdl = self.mdl.merge_and_unload()
        else:
            # Regular model loading
            self.mdl = ColIntern3_5.from_pretrained(
                model_name,
                device_map=self.device,
                torch_dtype=torch_dtype,
                adapter_kwargs={"revision": revision} if revision else {},
                **kwargs,
            )
        self.mdl.eval()
        
        # For local checkpoints, use the base model for the processor
        base_model_name = "OpenGVLab/InternVL3_5-1B-HF"
        processor_model_name = base_model_name if model_name.startswith("experiments/") else model_name
        
        # Load processor from base model
        self.processor = ColIntern3_5Processor.from_pretrained(processor_model_name)
        
        # Initialize processor kwargs (inherited from ColPaliEngineWrapper)
        self.processor_kwargs = {}

    def encode(self, sentences, **kwargs):
        return self.get_text_embeddings(texts=sentences, **kwargs)

    def encode_input(self, inputs):
        return self.mdl(**inputs)


# Training data for ColIntern3.5 (similar to ColPali training data)
COLINTERN3_5_TRAINING_DATA = {
    # Based on the same training set as ColPali models
    "DocVQA": ["train"],
    "InfoVQA": ["train"], 
    "TATDQA": ["train"],
    "arXivQA": ["train"],
}

# Model metadata for the trained ColIntern3.5 model
colintern3_5_1b_lora = ModelMeta(
    loader=partial(
        ColIntern3_5Wrapper,
        model_name="experiments/colintern3_5-1B-lora/checkpoint-1847",                                                    
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="local/colintern3_5-1B-lora",
    languages=["eng-Latn"],                                                  
    revision="checkpoint-1847",  # Our local checkpoint
    release_date="2025-09-05",  # Current date
    n_parameters=905_000_000,  # 905M parameters as shown in training                         
    memory_usage_mb=1800,  # Approximate memory usage in MB
    max_tokens=32768,  # InternVL3.5 context length
    embed_dim=3584,  # Based on InternVL3.5-1B hidden size
    license="apache-2.0",  # Must be lowercase
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],  # Only include valid frameworks
    reference="https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLINTERN3_5_TRAINING_DATA,
)
