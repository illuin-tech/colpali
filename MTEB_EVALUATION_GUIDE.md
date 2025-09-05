# ColIntern3.5 MTEB Evaluation Guide

This directory contains the MTEB wrapper and evaluation scripts for ColIntern3.5 models.

## Files

- `mteb_wrappers/colintern3_5_models.py` - MTEB wrapper for ColIntern3.5 models
- `scripts/test_mteb_integration.py` - Test script to verify MTEB integration
- `scripts/run_vidore_eval.py` - Main evaluation script for ViDoRe benchmarks
- `scripts/evaluate_colintern3_5.py` - Alternative evaluation script
- `scripts/simple_mteb_eval.py` - Simple evaluation example

## Quick Start

1. **Test the integration first:**
```bash
source .venv/bin/activate
python scripts/test_mteb_integration.py
```

2. **Run ViDoRe evaluation:**
```bash
source .venv/bin/activate
python scripts/run_vidore_eval.py --checkpoint-path experiments/colintern3_5-1B-lora/checkpoint-1847
```

3. **Run specific benchmarks:**
```bash
# Run only ViDoRe v1
python scripts/run_vidore_eval.py --benchmarks "ViDoRe(v1)"

# Run both v1 and v2
python scripts/run_vidore_eval.py --benchmarks "ViDoRe(v1)" "ViDoRe(v2)"
```

4. **Customize evaluation:**
```bash
python scripts/run_vidore_eval.py \
  --checkpoint-path experiments/colintern3_5-1B-lora/checkpoint-1847 \
  --benchmarks "ViDoRe(v1)" "ViDoRe(v2)" \
  --output-dir results/my_evaluation \
  --batch-size 8
```

## Command Line Interface (CLI)

You can also use the MTEB CLI directly:

```bash
# Install mteb if not already installed
pip install mteb

# Run evaluation using CLI
mteb run -b "ViDoRe(v1)" -m "local/colintern3_5-checkpoint-1847"
```

## Expected Output

The evaluation will create results in the specified output directory with:
- JSON files containing detailed metrics
- Summary statistics
- Per-task performance scores

Typical ViDoRe v1 tasks include:
- DocVQA
- InfoVQA
- TabFQuAD
- ArXivQA
- And other document retrieval tasks

## Performance Notes

- Evaluation can take several hours depending on benchmark size
- GPU recommended for faster inference
- Memory usage: ~1.8GB for the model
- Batch size can be adjusted based on available GPU memory

## Troubleshooting

If you encounter issues:

1. **Model loading errors**: Ensure the checkpoint path exists and contains the trained model
2. **Processor errors**: The wrapper automatically uses the base InternVL model for preprocessing
3. **MTEB errors**: Make sure you have the latest version of mteb installed
4. **CUDA errors**: Reduce batch size if you run out of GPU memory

## Expected Scores

Based on similar ColPali models, you can expect scores in the range:
- ViDoRe v1: 75-85 (depending on training quality)
- ViDoRe v2: 70-80 (harder benchmark)

Your model's performance will depend on the training data and hyperparameters used.
