# Useful info: 
# - GPU: RTX 5090 32GB
# - System RAM: 64GB
# - CUDA: 12.8
# - torch==2.8.0+cu128
# - nvidia-cuda-nvrtc-cu12==12.8.93
# - nvidia-cuda-runtime-cu12==12.8.90
# - flash_attn==2.8.3
# - accelerate==1.8.1
# - bitsandbytes==0.47.0
import argparse
import shutil
from pathlib import Path
import sys
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments
import multiprocessing as mp

# Ensure 'colpali/' directory is on sys.path so 'colpali_engine' is importable when running this file
_THIS_FILE = Path(__file__).resolve()
_COLPALI_DIR = _THIS_FILE.parents[3]  # .../colpali
if str(_COLPALI_DIR) not in sys.path:
    sys.path.insert(0, str(_COLPALI_DIR))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------- CRITICAL: avoid /dev/shm ----------
import torch.multiprocessing as torch_mp  # noqa: E402
try:
    mp.set_sharing_strategy("file_descriptor")  # no torch_shm_manager
except Exception:
    pass


from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import ColbertLoss, ColbertPairwiseCELoss
from colpali_engine.models import ColIntern3_5, ColIntern3_5Processor
from colpali_engine.trainer.colmodel_torch_training import ColModelTorchTraining
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_train_set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="./runs/8", help="where to write model + script copy")
    p.add_argument("--lr", type=float, default=5e-5, help="learning rate")  # keep your default
    p.add_argument("--tau", type=float, default=0.02, help="temperature for loss function")
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"], help="trainer to use")
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "pairwise"], help="loss function to use")
    p.add_argument("--peft", action="store_true", help="use PEFT for training")
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
            normalize_scores=False,
        )
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")

        # ---- InternVL3_5 backbone + processor ----
        # NOTE: InternVL3.5 uses 256 tokens per patch after pixel shuffle compression.
        # Default: 12 patches × 256 = 3072 tokens. We use 1536 (6 patches) for memory efficiency.
    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=ColIntern3_5Processor.from_pretrained(
            pretrained_model_name_or_path="OpenGVLab/InternVL3_5-1B-HF",
            max_num_visual_tokens=1536,  # 6 patches × 256 tokens = 1536 (memory efficient)
        ),
        model=ColIntern3_5.from_pretrained(
            pretrained_model_name_or_path="OpenGVLab/InternVL3_5-1B-HF",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # InternVL supports flash-attn:contentReference[oaicite:9]{index=9}
            trust_remote_code=True,                  # recommended for InternVL HF impl & weights:contentReference[oaicite:10]{index=10}
        ),

        # ---- Datasets identical to your flow ----
        train_dataset=load_train_set(),
        eval_dataset=ColPaliEngineDataset(
            load_dataset("./data_dir/colpali_train_set", split="test"), pos_target_column_name="image"
        ),
        run_eval=True,

        loss_func=loss_func,

        tr_args=TrainingArguments(
            output_dir=None,
            overwrite_output_dir=True,
            num_train_epochs=1,  # CRITICAL: Must be 5 for proper convergence
            per_device_train_batch_size=16,  # Optimal for RTX 5090 32GB
            gradient_accumulation_steps=4,   # Effective batch size = 64
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=16,   # Match training batch size
            eval_strategy="steps",
            dataloader_num_workers=4,       # Increase workers for faster data loading
            # dataloader_prefetch_factor=2,   # Prefetch batches for efficiency
            save_steps=500,               # Save more frequently
            logging_steps=10,
            eval_steps=100,                # Evaluate more frequently
            warmup_steps=100,              # Optimal warmup based on architecture analysis
            learning_rate=args.lr,         # Uses 5e-5 default (optimal)
            save_total_limit=1,            # Keep more checkpoints
            bf16=True,                     # Enable bfloat16 training to match model dtype
            dataloader_pin_memory=True,    # Disable pin memory to save GPU memory
            optim="adamw_bnb_8bit",        # Keep fused optimizer
            # remove_unused_columns=False, # Don't remove columns to avoid reprocessing
            # fp16_full_eval=False,        # Use bf16 for eval too
            tf32=True,                      # Enable TensorFloat-32 for faster computation
            report_to="wandb",              # Enable wandb logging
            torch_empty_cache_steps=4,     # Clear cache more frequently
        ),

        # ---- LoRA over language model + the custom projection head only ----
        # Your Qwen script targets MLP/attn projections and `custom_text_proj`, while excluding vision:contentReference[oaicite:11]{index=11}.
        # InternVL base model exposes `language_model` directly (no extra `.model` wrapper in the base class):contentReference[oaicite:12]{index=12},
        # so target that path instead:
        peft_config=LoraConfig(
            r=32,  # Reduced rank from 32 to 16 for memory efficiency
            lora_alpha=32,  # Adjusted alpha proportionally
            lora_dropout=0.1,
            init_lora_weights="gaussian",
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=[
                "language_model.layers.*.self_attn.q_proj",
                "language_model.layers.*.self_attn.k_proj", 
                "language_model.layers.*.self_attn.v_proj",
                "language_model.layers.*.self_attn.o_proj",
                "language_model.layers.*.mlp.gate_proj",
                "language_model.layers.*.mlp.up_proj",
                "language_model.layers.*.mlp.down_proj",
                "custom_text_proj",
            ],
        ) if args.peft else None,
    )

    # Save a copy of the script for provenance (same as your flow)
    # Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    # shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)

    trainer = ColModelTraining(config) if args.trainer == "hf" else ColModelTorchTraining(config)
    trainer.train()
    trainer.save()
