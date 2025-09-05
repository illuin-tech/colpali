# Scripts Directory

This directory contains utility scripts for training, evaluation, and optimization of ColIntern3.5 models.

## 🔧 Core Scripts

### Training Scripts
- **`configs/internvl3_5/train_colintern35_model.py`** - Main training script with optimal defaults
- **`hyperparameter_optimizer.py`** - Architecture-based hyperparameter calculation
- **`optuna_optimizer.py`** - Advanced hyperparameter search with Optuna

### Evaluation Scripts  
- **`evaluate_colintern3_5.py`** - Comprehensive model evaluation
- **`run_vidore_eval.py`** - ViDoRe benchmark evaluation

### Verification Scripts
- **`verify_target_modules.py`** - Validate LoRA target modules for your model architecture

### Other Scripts
- **`api_call.py`** - API interaction utilities
- **`compute_hardnegs.py`** - Hard negative mining
- **`reasoning_queries.py`** - Query reasoning analysis
- **`test_mteb_integration.py`** - MTEB framework testing

## 🎯 Key Script: verify_target_modules.py

This script validates that your LoRA target modules are correctly configured for InternVL3.5-1B-HF.

### Usage
```bash
cd /path/to/colpali
python scripts/verify_target_modules.py
```

### What It Checks
- ✅ All target modules exist in the model
- ✅ Correct module paths and naming
- ✅ Proper layer coverage (all 28 language model layers)
- ✅ Critical components included (custom_text_proj)
- ✅ Vision components correctly excluded

### Expected Output
```
================================================================================
LORA TARGET MODULE VERIFICATION FOR InternVL3.5-1B-HF
================================================================================

📍 Target: language_model.layers.*.self_attn.q_proj
  ✅ Found 28 matches
     Covers layers 0-27 (28 total layers)

[... continues for all target modules ...]

================================================================================
SUMMARY
================================================================================
✅ Total modules that will receive LoRA adapters: 197
📊 Language model layers: 28
📊 Expected total modules: 197 = 197
✅ Module count matches expectation!
```

### When to Run
- Before starting any training
- When switching model architectures
- When debugging poor training performance
- When modifying target_modules configuration

## 🚀 Quick Start Workflow

```bash
# 1. Verify target modules
python scripts/verify_target_modules.py

# 2. Get optimal hyperparameters
python scripts/hyperparameter_optimizer.py

# 3. Start training
python scripts/configs/internvl3_5/train_colintern35_model.py \
  --output-dir ./experiments/colintern3_5-optimal \
  --peft

# 4. Evaluate results
python scripts/evaluate_colintern3_5.py \
  --model-path ./experiments/colintern3_5-optimal
```

## 📊 Hyperparameter Optimization

### Architecture-Based Optimization
```bash
# Get scientifically-determined optimal parameters
python scripts/hyperparameter_optimizer.py
```

**Features:**
- Model architecture analysis (parameter count, hidden size, layers)
- Memory usage estimation
- Optimal batch size calculation
- Learning rate scaling based on model size
- LoRA rank recommendations

### Advanced Optuna Search
```bash
# Automated hyperparameter search
python scripts/optuna_optimizer.py --trials 20
```

**Features:**
- Multi-objective optimization
- Bayesian parameter search
- Early stopping for poor trials
- Results saved to `optuna_best_params.json`

## 🔍 Troubleshooting Scripts

If you encounter training issues, these scripts can help diagnose problems:

1. **Target Module Issues**: `python scripts/verify_target_modules.py`
2. **Hyperparameter Problems**: `python scripts/hyperparameter_optimizer.py`
3. **Evaluation Issues**: `python scripts/evaluate_colintern3_5.py --debug`

## 📝 Script Configuration

Most scripts use command-line arguments for configuration:

```bash
# Example: Training with custom parameters
python scripts/configs/internvl3_5/train_colintern35_model.py \
  --output-dir ./experiments/custom-run \
  --lr 3e-5 \
  --tau 0.01 \
  --peft

# Example: Evaluation with specific benchmarks
python scripts/evaluate_colintern3_5.py \
  --model-path ./experiments/custom-run \
  --batch-size 32 \
  --benchmarks VidoreDocVQARetrieval
```

For detailed usage information, run any script with `--help`:
```bash
python scripts/train_colintern35_model.py --help
```
