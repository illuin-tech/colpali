import argparse
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import ColbertLoss, ColbertPairwiseCELoss
from colpali_engine.models import ColInternVL3_5, ColInternVL3_5_Processor
from colpali_engine.trainer.colmodel_torch_training import ColModelTorchTraining
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_train_set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=float, default=0.02, help="temperature for loss function")
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"], help="trainer to use")
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "pairwise"], help="loss function to use")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    loss_func = ColbertLoss(
        temperature=args.tau,
        normalize_scores=True,
        use_smooth_max=False,
        pos_aware_negative_filtering=False,
    )

    config = ColModelTrainingConfig(
        output_dir="models/colinternvl3_5_2b",
        processor=ColInternVL3_5_Processor.from_pretrained(
            pretrained_model_name_or_path="./models/base_models/colinternvl3_5-2b-base",
            max_num_visual_tokens=768,
        ),
        model=ColInternVL3_5.from_pretrained(
            pretrained_model_name_or_path="./models/base_models/colinternvl3_5-2b-base",
            dtype=torch.bfloat16,
            # low_cpu_mem_usage=True,
            # use_cache=False,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ),
        train_dataset=load_train_set(),
        eval_dataset=ColPaliEngineDataset(
            load_dataset("./data_dir/colpali_train_set", split="test"), pos_target_column_name="image"
        ),
        run_eval=True,
        loss_func=loss_func,
        tr_args=TrainingArguments(
            output_dir=None,
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_device_train_batch_size=32,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=16,
            eval_strategy="steps",
            dataloader_num_workers=8,
            save_steps=500,
            logging_steps=10,
            eval_steps=100,
            warmup_steps=100,
            learning_rate=2e-05,
            save_total_limit=1,
            ddp_find_unused_parameters=False,
            # run_name="visual-colinternvl3_5-2b-test",
            report_to=None,
        ),
        peft_config=LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
            init_lora_weights="gaussian",
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules="(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
        )
    )

    # make sure output_dir exists and copy script for provenance
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)

    trainer = ColModelTraining(config) if args.trainer == "hf" else ColModelTorchTraining(config)
    trainer.train()
    trainer.save()
