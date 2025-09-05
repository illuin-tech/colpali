#!/usr/bin/env python3
"""
Programmatic hyperparameter optimization for ColIntern3.5 based on model architecture.
"""

import math
import torch
from transformers import AutoConfig


def analyze_model_architecture(model_name="OpenGVLab/InternVL3_5-1B-HF"):
    """Analyze model architecture to determine optimal hyperparameters."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Get model parameters without loading full model
    # Estimate parameters from config
    text_params = (
        config.text_config.hidden_size * config.text_config.vocab_size +  # embeddings
        config.text_config.num_hidden_layers * (
            4 * config.text_config.hidden_size * config.text_config.intermediate_size +  # MLP
            4 * config.text_config.hidden_size * config.text_config.hidden_size  # attention
        )
    )
    
    vision_params = 1e8  # Rough estimate for vision encoder (100M)
    total_params = text_params + vision_params + 128 * config.text_config.hidden_size  # custom_text_proj
    
    # Architecture analysis
    analysis = {
        "model_size": total_params / 1e6,  # In millions
        "hidden_size": config.text_config.hidden_size,
        "num_layers": config.text_config.num_hidden_layers,
        "vocab_size": config.text_config.vocab_size,
        "vision_hidden_size": getattr(config.vision_config, 'hidden_size', 1024),
        "trainable_params": 0.037,  # LoRA is ~0.037M trainable params
    }
    
    return analysis, config


def compute_optimal_hyperparameters(analysis, gpu_memory_gb=32):
    """Compute optimal hyperparameters based on architecture analysis."""
    
    model_size_mb = analysis["model_size"]
    hidden_size = analysis["hidden_size"]
    
    # 1. Learning Rate (based on model size and hidden dimension)
    # Smaller models need higher LR, larger models need lower LR
    if model_size_mb < 1000:  # < 1B parameters
        base_lr = 5e-5
    elif model_size_mb < 3000:  # 1-3B parameters
        base_lr = 3e-5
    else:  # > 3B parameters
        base_lr = 1e-5
    
    # Scale by hidden size (larger hidden size needs smaller LR)
    lr_scale = math.sqrt(1024 / hidden_size)  # Normalize to 1024 baseline
    optimal_lr = base_lr * lr_scale
    
    # 2. Batch Size (based on GPU memory and model size)
    # Estimate memory usage: model + gradients + activations
    model_memory = model_size_mb * 4  # FP32 equivalent
    gradient_memory = model_memory * 2  # Gradients + optimizer states
    
    # Available memory for activations
    available_memory = (gpu_memory_gb * 1024 - model_memory - gradient_memory) * 0.7  # 70% utilization
    
    # Estimate memory per sample (vision + text tokens)
    tokens_per_sample = 1536 + 256  # Visual tokens + text tokens
    memory_per_sample = tokens_per_sample * hidden_size * 4 / (1024 * 1024)  # MB
    
    max_batch_size = max(1, int(available_memory / memory_per_sample))
    
    # Optimal batch size for training stability (power of 2, but not too large)
    optimal_batch_size = min(32, 2 ** int(math.log2(max_batch_size)))
    
    # 3. Gradient Accumulation (to reach effective batch size of 64-128)
    target_effective_batch = 64
    gradient_accumulation = max(1, target_effective_batch // optimal_batch_size)
    
    # 4. Number of Epochs (based on model complexity)
    # More complex models need more epochs
    complexity_factor = math.log10(model_size_mb) / math.log10(1000)  # Normalize to 1B baseline
    optimal_epochs = max(3, min(8, int(3 + 2 * complexity_factor)))
    
    # 5. Warmup Steps (10% of first epoch)
    # Estimate dataset size (ColPali typically has ~127k samples)
    estimated_dataset_size = 127000
    steps_per_epoch = estimated_dataset_size // (optimal_batch_size * gradient_accumulation)
    warmup_steps = max(100, steps_per_epoch // 10)
    
    # 6. LoRA Rank (based on model size)
    if model_size_mb < 1000:
        lora_rank = 16
    elif model_size_mb < 3000:
        lora_rank = 32
    else:
        lora_rank = 64
    
    return {
        "learning_rate": optimal_lr,
        "per_device_train_batch_size": optimal_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "num_train_epochs": optimal_epochs,
        "warmup_steps": warmup_steps,
        "lora_rank": lora_rank,
        "effective_batch_size": optimal_batch_size * gradient_accumulation,
        "estimated_training_time_hours": (steps_per_epoch * optimal_epochs) / 200,  # Rough estimate
    }


def generate_grid_search_configs(base_config, search_space=None):
    """Generate configurations for grid search."""
    if search_space is None:
        search_space = {
            "learning_rate": [base_config["learning_rate"] * x for x in [0.5, 1.0, 2.0]],
            "lora_rank": [16, 32, 64],
            "warmup_ratio": [0.05, 0.1, 0.15],
        }
    
    configs = []
    for lr in search_space["learning_rate"]:
        for rank in search_space["lora_rank"]:
            for warmup_ratio in search_space["warmup_ratio"]:
                config = base_config.copy()
                config.update({
                    "learning_rate": lr,
                    "lora_rank": rank,
                    "warmup_steps": int(config["warmup_steps"] * warmup_ratio / 0.1),
                    "config_name": f"lr{lr:.0e}_rank{rank}_warmup{warmup_ratio:.2f}"
                })
                configs.append(config)
    
    return configs


def print_recommendations(analysis, hyperparams):
    """Print detailed recommendations."""
    print("=" * 80)
    print("🧠 MODEL ARCHITECTURE ANALYSIS")
    print("=" * 80)
    print(f"Model Size: {analysis['model_size']:.1f}M parameters")
    print(f"Hidden Size: {analysis['hidden_size']}")
    print(f"Number of Layers: {analysis['num_layers']}")
    print(f"Trainable Parameters: {analysis['trainable_params']:.1f}M")
    
    print("\n" + "=" * 80)
    print("⚡ OPTIMAL HYPERPARAMETERS")
    print("=" * 80)
    print(f"Learning Rate: {hyperparams['learning_rate']:.2e}")
    print(f"Batch Size: {hyperparams['per_device_train_batch_size']}")
    print(f"Gradient Accumulation: {hyperparams['gradient_accumulation_steps']}")
    print(f"Effective Batch Size: {hyperparams['effective_batch_size']}")
    print(f"Number of Epochs: {hyperparams['num_train_epochs']}")
    print(f"Warmup Steps: {hyperparams['warmup_steps']}")
    print(f"LoRA Rank: {hyperparams['lora_rank']}")
    
    print(f"\n📊 Estimated Training Time: {hyperparams['estimated_training_time_hours']:.1f} hours")
    
    print("\n" + "=" * 80)
    print("🚀 TRAINING COMMAND")
    print("=" * 80)
    cmd = f"""python scripts/configs/internvl3_5/train_colintern35_model.py \\
  --output-dir ./experiments/colintern3_5-optimized \\
  --lr {hyperparams['learning_rate']:.2e} \\
  --peft"""
    print(cmd)


if __name__ == "__main__":
    print("🔍 Analyzing InternVL3.5 architecture...")
    analysis, config = analyze_model_architecture()
    
    print("💡 Computing optimal hyperparameters...")
    hyperparams = compute_optimal_hyperparameters(analysis)
    
    print_recommendations(analysis, hyperparams)
    
    # Generate grid search if requested
    print("\n" + "=" * 80)
    print("🔬 GRID SEARCH CONFIGURATIONS")
    print("=" * 80)
    grid_configs = generate_grid_search_configs(hyperparams)
    print(f"Generated {len(grid_configs)} configurations for hyperparameter search")
    for i, config in enumerate(grid_configs[:3]):  # Show first 3
        print(f"Config {i+1}: {config['config_name']}")
        print(f"  LR: {config['learning_rate']:.2e}, Rank: {config['lora_rank']}, Warmup: {config['warmup_steps']}")
