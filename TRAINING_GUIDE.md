# ColIntern3.5 Training & Evaluation Guide

This guide provides comprehensive instructions for training and evaluating ColIntern3.5 models using the ColPali framework with proper hyperparameter optimization.

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Training Instructions](#training-instructions)
- [Evaluation Instructions](#evaluation-instructions)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Troubleshooting](#troubleshooting)
- [Performance Expectations](#performance-expectations)

## 🖥️ System Requirements

### Minimum Hardware
- **GPU**: 16GB VRAM (RTX 4080, RTX 4090, A100, etc.)
- **RAM**: 32GB system memory
- **Storage**: 100GB free space for datasets and checkpoints

### Recommended Hardware
- **GPU**: RTX 5090 32GB or A100 40GB/80GB
- **RAM**: 64GB+ system memory
- **Storage**: 500GB NVMe SSD

### Software Requirements
```bash
# Python packages (automatically installed with uv/pip)
torch>=2.8.0
transformers>=4.46.0
flash-attn>=2.8.0
colpali-engine
peft>=0.12.0
wandb  # for experiment tracking
optuna  # for hyperparameter optimization
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and setup environment
cd /path/to/colpali
source .venv/bin/activate  # or use uv

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Run Hyperparameter Analysis
```bash
# Get scientifically-optimized hyperparameters for your architecture
python scripts/hyperparameter_optimizer.py
```

### 3. Start Training with Optimal Settings
```bash
# Recommended training command
python scripts/configs/internvl3_5/train_colintern35_model.py \
  --output-dir ./experiments/colintern3_5-optimal \
  --lr 5e-5 \
  --peft
```

### 4. Evaluate Trained Model
```bash
# Evaluate on ViDoRe benchmarks
python scripts/evaluate_colintern3_5.py \
  --model-path ./experiments/colintern3_5-optimal \
  --batch-size 16
```

## 🏋️ Training Instructions

### Core Training Script
The main training script is located at `scripts/configs/internvl3_5/train_colintern35_model.py`.

### Critical Training Parameters

#### ✅ **Optimal Settings (Expected NDCG@5: 0.60-0.80)**
```bash
python scripts/configs/internvl3_5/train_colintern35_model.py \
  --output-dir ./experiments/colintern3_5-production \
  --lr 5e-5 \
  --peft
```

**Internal Parameters (in script):**
- `num_train_epochs=5` ⚠️ **CRITICAL: Must be 5, not 1**
- `per_device_train_batch_size=16`
- `gradient_accumulation_steps=4` (Effective batch size: 64)
- `learning_rate=5e-5`
- `lora_rank=32`
- `warmup_steps=100`

#### ❌ **Poor Settings (Results in NDCG@5: ~0.11)**
```bash
# DON'T DO THIS - These settings cause poor performance
python scripts/configs/internvl3_5/train_colintern35_model.py \
  --output-dir ./experiments/bad-training \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --eval_strategy no \
  --dataloader_num_workers 0
```

### Training Configuration Details

#### **Batch Size Guidelines**
| GPU Memory | Recommended Batch Size | Gradient Accumulation | Effective Batch Size |
|------------|----------------------|---------------------|-------------------|
| 16GB       | 8                    | 8                   | 64                |
| 24GB       | 12                   | 5-6                 | 64                |
| 32GB       | 16                   | 4                   | 64                |
| 40GB+      | 24-32                | 2-3                 | 64                |

#### **Learning Rate Scaling**
- **Base Model (1B params)**: `5e-5`
- **Larger Models (3B+ params)**: `3e-5`
- **Smaller Models (<500M params)**: `7e-5`

#### **LoRA Configuration**
```python
LoraConfig(
    r=32,                    # Rank: 16 for <1B, 32 for 1B, 64 for >3B
    lora_alpha=32,           # Keep equal to rank
    lora_dropout=0.1,        # Standard dropout
    target_modules=[
        "language_model.layers.*.self_attn.q_proj",
        "language_model.layers.*.self_attn.k_proj", 
        "language_model.layers.*.self_attn.v_proj",
        "language_model.layers.*.self_attn.o_proj",
        "language_model.layers.*.mlp.gate_proj",
        "language_model.layers.*.mlp.up_proj",
        "language_model.layers.*.mlp.down_proj",
        "custom_text_proj",     # CRITICAL: Include projection layer
    ],
)
```

### Monitoring Training

#### **WandB Integration**
Training automatically logs to Weights & Biases:
```bash
# View training progress
wandb login
# Logs appear at: https://wandb.ai/your-username/huggingface
```

#### **Key Metrics to Monitor**
- **Training Loss**: Should decrease from ~2.4 to ~0.8-1.0
- **Eval Loss**: Should follow training loss closely
- **Learning Rate**: Should follow warmup then decay schedule
- **Grad Norm**: Should be stable (~1-3), not oscillating wildly

### Expected Training Time
| Model Size | GPU        | Epochs | Estimated Time |
|------------|------------|--------|----------------|
| 1B         | RTX 5090   | 5      | 30-40 hours    |
| 1B         | A100 40GB  | 5      | 20-25 hours    |
| 1B         | A100 80GB  | 5      | 15-20 hours    |

## 📊 Evaluation Instructions

### Standard Evaluation
```bash
# Evaluate on all ViDoRe benchmarks
python scripts/evaluate_colintern3_5.py \
  --model-path ./experiments/colintern3_5-optimal \
  --batch-size 16 \
  --benchmarks "ViDoRe(v1)" "ViDoRe(v2)"
```

### Single Benchmark Evaluation
```bash
# Test on specific benchmark
python scripts/evaluate_colintern3_5.py \
  --model-path ./experiments/colintern3_5-optimal \
  --batch-size 16 \
  --benchmarks VidoreDocVQARetrieval
```

### Custom Evaluation Script
```bash
# Alternative evaluation script
python scripts/run_vidore_eval.py \
  --checkpoint-path ./experiments/colintern3_5-optimal \
  --batch-size 16
```

### Evaluation Metrics

#### **Primary Metrics**
- **NDCG@5**: Normalized Discounted Cumulative Gain at 5
- **NDCG@10**: NDCG at 10 results
- **Recall@100**: Percentage of relevant docs in top 100

#### **Performance Expectations**
| Model Quality | NDCG@5 Range | Status |
|---------------|--------------|--------|
| Excellent     | 0.70-0.85    | 🎯 Target performance |
| Good          | 0.50-0.70    | ✅ Acceptable |
| Poor          | 0.10-0.30    | ⚠️ Check hyperparameters |
| Broken        | <0.10        | ❌ Training failed |

## 🔧 Hyperparameter Optimization

### Automated Architecture Analysis
```bash
# Get optimal hyperparameters for your model architecture
python scripts/hyperparameter_optimizer.py
```

**Output Example:**
```
================================================================================
⚡ OPTIMAL HYPERPARAMETERS
================================================================================
Learning Rate: 5.00e-05
Batch Size: 16
Gradient Accumulation: 4
Effective Batch Size: 64
Number of Epochs: 5
Warmup Steps: 198
LoRA Rank: 32
```

### Optuna-Based Search (Advanced)
```bash
# Run automated hyperparameter search with Optuna
python scripts/optuna_optimizer.py --trials 10

# Results saved to: optuna_best_params.json
```

### Manual Hyperparameter Guidelines

#### **Learning Rate Selection**
```python
# Based on model size
def get_optimal_lr(model_size_mb):
    if model_size_mb < 1000:      # <1B parameters
        return 5e-5
    elif model_size_mb < 3000:    # 1-3B parameters  
        return 3e-5
    else:                         # >3B parameters
        return 1e-5
```

#### **Batch Size Calculation**
```python
# Based on GPU memory
def get_optimal_batch_size(gpu_memory_gb):
    memory_per_sample = 150  # MB (rough estimate)
    available_memory = gpu_memory_gb * 1024 * 0.7  # 70% utilization
    max_batch_size = int(available_memory / memory_per_sample)
    return min(32, 2 ** int(math.log2(max_batch_size)))
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### **Issue: Poor Performance (NDCG@5 < 0.30)**
**Symptoms:**
- Low evaluation scores
- Model seems undertrained

**Solutions:**
1. **Check hyperparameters:**
   ```bash
   # Verify you're using optimal settings
   python scripts/hyperparameter_optimizer.py
   ```

2. **Increase training epochs:**
   ```python
   # In train_colintern35_model.py
   num_train_epochs=5  # Not 1!
   ```

3. **Fix batch size:**
   ```python
   per_device_train_batch_size=16  # Not 2!
   gradient_accumulation_steps=4
   ```

#### **Issue: PEFT Weights Not Loading**
**Symptoms:**
- Warning: "Some weights were not initialized"
- Poor performance despite training

**Solution:**
The model wrapper automatically handles PEFT loading with `merge_and_unload()`. Verify with:
```python
from mteb_wrappers.colintern3_5_models import ColIntern3_5Wrapper
model = ColIntern3_5Wrapper('path/to/checkpoint')
# Check if weights look trained (not random)
custom_proj = model.mdl.custom_text_proj
print(f"Weight range: [{custom_proj.weight.min():.6f}, {custom_proj.weight.max():.6f}]")
```

#### **Issue: Out of Memory Errors**
**Solutions:**
1. **Reduce batch size:**
   ```python
   per_device_train_batch_size=8  # Reduce from 16
   gradient_accumulation_steps=8  # Increase to maintain effective batch size
   ```

2. **Enable gradient checkpointing:**
   ```python
   gradient_checkpointing=True
   ```

3. **Reduce sequence length:**
   ```python
   max_num_visual_tokens=1024  # Reduce from 1536
   ```

#### **Issue: Training Stalls or NaN Loss**
**Solutions:**
1. **Check learning rate:**
   ```python
   learning_rate=2e-5  # Reduce if training is unstable
   ```

2. **Adjust warmup:**
   ```python
   warmup_steps=200  # Increase warmup period
   ```

3. **Check gradient clipping:**
   ```python
   max_grad_norm=1.0  # Add gradient clipping
   ```

### Performance Debugging Checklist

- [ ] **Epochs ≥ 5**: Essential for convergence
- [ ] **Effective batch size ≥ 64**: Required for stable gradients
- [ ] **Learning rate = 5e-5**: Optimal for 1B parameter models
- [ ] **PEFT enabled**: `--peft` flag included
- [ ] **Target modules include custom_text_proj**: Critical for performance
- [ ] **BF16 training enabled**: Matches model precision
- [ ] **Flash attention enabled**: Improves efficiency

## 📈 Performance Expectations

### Benchmark Performance Targets

| Benchmark | Expected NDCG@5 | Status |
|-----------|----------------|--------|
| VidoreArxivQARetrieval | 0.60-0.75 | 🎯 |
| VidoreDocVQARetrieval | 0.65-0.80 | 🎯 |
| VidoreInfoVQARetrieval | 0.55-0.70 | 🎯 |
| VidoreTatdqaRetrieval | 0.50-0.65 | 🎯 |

### Training Progress Indicators

#### **Healthy Training:**
```
Epoch 1: Loss 2.40 → 1.50
Epoch 2: Loss 1.50 → 1.20  
Epoch 3: Loss 1.20 → 1.00
Epoch 4: Loss 1.00 → 0.85
Epoch 5: Loss 0.85 → 0.75
```

#### **Problematic Training:**
```
Epoch 1: Loss 2.40 → 2.35  # Too slow convergence
# OR
Epoch 1: Loss 2.40 → NaN   # Learning rate too high
# OR  
Epoch 1: Loss oscillating wildly  # Batch size too small
```

### Resource Usage Guidelines

| Component | Normal Usage | Warning Signs |
|-----------|-------------|---------------|
| GPU Memory | 80-90% | >95% (OOM risk) |
| GPU Utilization | 90-100% | <70% (inefficient) |
| Training Speed | 2-4 sec/step | >10 sec/step |
| CPU Memory | <80% | >90% |

## 📚 Additional Resources

### Useful Commands
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training progress
tail -f experiments/*/trainer_state.json

# Compare model performance
python scripts/compare_models.py model1 model2

# Generate training report
python scripts/training_report.py --model-path experiments/model
```

### File Structure
```
experiments/
├── colintern3_5-optimal/          # Recommended output directory
│   ├── adapter_config.json        # PEFT configuration
│   ├── adapter_model.safetensors  # Trained weights
│   ├── trainer_state.json         # Training progress
│   └── checkpoint-*/               # Intermediate checkpoints
results/
├── mteb_evaluation/                # Evaluation results
│   └── model_name/
│       ├── VidoreDocVQARetrieval.json
│       └── ...
scripts/
├── hyperparameter_optimizer.py    # Architecture analysis
├── optuna_optimizer.py            # Advanced optimization
└── evaluate_colintern3_5.py      # Evaluation script
```

---

## 🎯 Summary: The Path to 80+ Performance

1. **Use the hyperparameter optimizer**: `python scripts/hyperparameter_optimizer.py`
2. **Train with optimal settings**: 5 epochs, batch size 16×4, learning rate 5e-5
3. **Monitor training closely**: Loss should decrease smoothly to ~0.8
4. **Evaluate properly**: Use the fixed model wrapper that loads PEFT weights correctly
5. **Expect 60-80 NDCG@5**: With proper hyperparameters, you should match ColSmol-500M performance

**The key insight**: Hyperparameters aren't just about speed—they fundamentally determine whether your model learns effectively or not. Small changes (batch size 2→16, epochs 1→5) can mean the difference between 11% and 80% performance!
