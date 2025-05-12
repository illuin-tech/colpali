import argparse
import shutil
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import TrainingArguments

from colpali_engine.loss.late_interaction_losses import ColbertLoss, ColbertPairwiseCELoss
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.trainer.colmodel_torch_training import ColModelTorchTraining
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_train_set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True, help="where to write model + script copy")
    p.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    p.add_argument("--tau", type=float, default=0.02, help="temperature for loss function")
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"], help="trainer to use")
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "pairwise"], help="loss function to use")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.loss == "ce":
        loss_func = ColbertLoss(
            temperature=args.tau,
            normalize_scores=True,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
        )
    elif args.loss == "pairwise":
        loss_func = ColbertPairwiseCELoss(
            normalize_scores=True,
        )
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")
    
    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=ColQwen2Processor.from_pretrained(
            pretrained_model_name_or_path="./models/base_models/colqwen2-base",
            max_num_visual_tokens=1024,
        ),
        model=ColQwen2.from_pretrained(
            pretrained_model_name_or_path="./models/base_models/colqwen2-base",
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation="flash_attention_2",
        ),
        dataset_loading_func=load_train_set,
        run_eval=True,
        loss_func=loss_func,
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
            learning_rate=args.lr,
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

    # make sure output_dir exists and copy script for provenance
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)

    trainer = ColModelTraining(config) if args.trainer == "hf" else ColModelTorchTraining(config)
    trainer.train()
    trainer.save()
