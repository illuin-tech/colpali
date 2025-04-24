import os
import shutil
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import TrainingArguments

from colpali_engine.loss.late_interaction_losses import ColbertLoss
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_train_set_ir_negs, load_train_set

config = ColModelTrainingConfig(
    output_dir="./models/colqwen2-hardneg-5e-2404",
    processor=ColQwen2Processor.from_pretrained(
        pretrained_model_name_or_path="./models/base_models/colqwen2-base",
    ),
    model=ColQwen2.from_pretrained(
        pretrained_model_name_or_path="./models/base_models/colqwen2-base",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2",
    ),
    dataset_loading_func=load_train_set_ir_negs,
    eval_dataset_loader=None,
    run_eval=True,
    loss_func=ColbertLoss(normalize_scores=True),
    tr_args=TrainingArguments(
        output_dir=None,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=64,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_eval_batch_size=16,
        eval_strategy="steps",
        dataloader_num_workers=2,
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
        target_modules="(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$)",
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
