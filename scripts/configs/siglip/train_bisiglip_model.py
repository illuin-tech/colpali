import os
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.models import BiSiglip, BiSiglipProcessor
from colpali_engine.loss.bi_encoder_losses import BiEncoderLoss
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_train_set

config = ColModelTrainingConfig(
    output_dir="./models/bisiglip-base-ce-0810",
    processor=BiSiglipProcessor.from_pretrained(
        pretrained_model_name_or_path="./models/base_models/siglip2-base-patch16-512",
    ),
    model=BiSiglip.from_pretrained(
        pretrained_model_name_or_path="./models/base_models/siglip2-base-patch16-512",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ),
    train_dataset=load_train_set(),  # load_train_set_ir_negs,
    eval_dataset=ColPaliEngineDataset(
        load_dataset("./data_dir/colpali_train_set", split="test"), pos_target_column_name="image"
    ),
    run_eval=True,
    loss_func=BiEncoderLoss(),  # BiNegativeCELoss(in_batch_term_weight=0.5),
    # loss_func=BiSigLipEncoderLoss(),
    tr_args=TrainingArguments(
        output_dir=None,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=64,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_eval_batch_size=16,
        eval_strategy="steps",
        dataloader_num_workers=4,
        save_steps=500,
        logging_steps=10,
        eval_steps=100,
        warmup_steps=100,
        learning_rate=2e-4,
        save_total_limit=1,
    ),
    peft_config=LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        init_lora_weights="gaussian",
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules="((.*(text_model).*(k_proj|q_proj|v_proj|out_proj).*$)|logit_scale|logit_bias)",  # noqa: E501,
    ),
)


if __name__ == "__main__":
    # ensure output_dir exists
    os.makedirs(config.output_dir, exist_ok=True)
    # version this script by copying it into the output dir
    current_script = Path(__file__)
    shutil.copy(current_script, Path(config.output_dir) / current_script.name)

    training_app = ColModelTraining(config)

    training_app.train()
    training_app.save()
