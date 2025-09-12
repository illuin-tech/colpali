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
import re
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


def get_target_modules_by_regex(model):
    """
    Generate target modules for LoRA using regex pattern (like Qwen2.5 approach).
    Pattern matches:
    - language_model layers (excluding visual): down_proj, gate_proj, up_proj, k_proj, q_proj, v_proj, o_proj
    - custom_text_proj
    """
    pattern = r"(.*(language_model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)"
    
    all_modules = dict(model.named_modules())
    target_modules = []
    
    for name, module in all_modules.items():
        if hasattr(module, 'weight') and re.match(pattern, name):
            target_modules.append(name)
    
    print(f"🎯 Regex pattern matched {len(target_modules)} target modules for LoRA:")
    print(f"   Pattern: {pattern}")
    print(f"   Expected: 197 modules (28 layers × 7 modules + 1 custom_text_proj)")
    print(f"   Matched: {len(target_modules)} modules")
    
    # Show sample matches
    if target_modules:
        print(f"   Sample modules:")
        for i, mod in enumerate(target_modules[:5]):
            print(f"     {i+1}. {mod}")
        if len(target_modules) > 5:
            print(f"     ... and {len(target_modules)-5} more")
    
    return target_modules


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="./runs/9", help="where to write model + script copy")
    p.add_argument("--lr", type=float, default=5e-5, help="learning rate")
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

    # Create model first to determine target modules
    base_model = ColIntern3_5.from_pretrained(
        pretrained_model_name_or_path="OpenGVLab/InternVL3_5-1B-HF",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # Generate target modules using regex pattern (cleaner than explicit enumeration)
    target_modules = get_target_modules_by_regex(base_model) if args.peft else None

    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=ColIntern3_5Processor.from_pretrained(
            pretrained_model_name_or_path="OpenGVLab/InternVL3_5-1B-HF",
            max_num_visual_tokens=3072,
        ),
        model=base_model,

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
            num_train_epochs=1,  
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=16,
            eval_strategy="steps",
            dataloader_num_workers=4,     
            dataloader_prefetch_factor=2,
            dataloader_persistent_workers=False,
            dataloader_pin_memory=False,    
            save_steps=500,               
            logging_steps=10,
            eval_steps=100,                
            warmup_steps=100,              
            learning_rate=args.lr,         
            save_total_limit=3,            
            bf16=True,                                 
            optim="adamw_bnb_8bit",        
            tf32=True,                     
            report_to="wandb",             
            torch_empty_cache_steps=4,     
        ),

        # ---- LoRA with base model parameters (like ColQwen2.5-base) ----
        peft_config=LoraConfig(
            r=32,                          # 🎯 Standard rank for base model training
            lora_alpha=32,                 # 🎯 Equal alpha for aggressive adaptation (1.0 ratio)
            lora_dropout=0.1,              # 🎯 Standard dropout for base model
            init_lora_weights="gaussian",
            bias="none", 
            task_type="FEATURE_EXTRACTION",
            target_modules=target_modules,  # 🎯 Regex-based targeting, cleaner than explicit enumeration
        ) if args.peft else None,
    )

    # Verify LoRA setup before training
    if args.peft:
        print(f"🔧 LoRA Configuration (Base Model Training):")
        print(f"  Target modules: {len(target_modules)}")
        print(f"  Expected trainable parameters: ~{len(target_modules) * 32 * 32:,}")
        print(f"  r={32}, alpha={32}, dropout={0.1}")
        print(f"  Alpha/Rank ratio: {32/32:.1f} (aggressive for base model training)")  

    trainer = ColModelTraining(config) if args.trainer == "hf" else ColModelTorchTraining(config)
    trainer.train()
    trainer.save()