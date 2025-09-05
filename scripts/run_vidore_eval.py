#!/usr/bin/env python3
"""
Run ViDoRe benchmark evaluation for ColIntern3.5 model.

Usage:
    python scripts/run_vidore_eval.py [--checkpoint-path PATH] [--benchmarks BENCH1 BENCH2] [--output-dir DIR]
"""

import argparse
import os
import sys
from pathlib import Path
from functools import partial

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mteb
from mteb.model_meta import ModelMeta
from mteb_wrappers.colintern3_5_models import ColIntern3_5Wrapper

def main():
    parser = argparse.ArgumentParser(description="Run ViDoRe evaluation for ColIntern3.5")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="experiments/colintern3_5-1B-lora/checkpoint-1847",
        help="Path to the ColIntern3.5 checkpoint"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["ViDoRe(v1)", "ViDoRe(v2)"],
        help="Benchmarks to evaluate on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/vidore_evaluation",
        help="Directory to save results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint path {args.checkpoint_path} does not exist!")
        sys.exit(1)
    
    print("🚀 Starting ViDoRe evaluation for ColIntern3.5")
    print(f"📍 Checkpoint: {args.checkpoint_path}")
    print(f"📊 Benchmarks: {args.benchmarks}")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"🔢 Batch size: {args.batch_size}")
    print("-" * 60)
    
    # Create model metadata
    model_name = f"local/colintern3_5-{Path(args.checkpoint_path).name}"
    
    custom_model_meta = ModelMeta(
        loader=partial(
            ColIntern3_5Wrapper,
            model_name=args.checkpoint_path
        ),
        name=model_name,
        modalities=["image", "text"],
        framework=["ColPali"],
        similarity_fn_name="max_sim",
        languages=["eng-Latn"],
        revision=Path(args.checkpoint_path).name,
        release_date="2025-09-05",
        n_parameters=905_000_000,
        memory_usage_mb=1800,
        max_tokens=32768,
        embed_dim=3584,
        license="apache-2.0",
        open_weights=True,
        public_training_code="https://github.com/illuin-tech/colpali",
        public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
        reference="https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF",
        use_instructions=True,
        training_datasets={"DocVQA": ["train"], "InfoVQA": ["train"], "TATDQA": ["train"], "arXivQA": ["train"]},
    )
    
    # Load model
    print("🔄 Loading model...")
    try:
        model = custom_model_meta.load_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Load benchmarks
    print(f"🔄 Loading benchmarks: {args.benchmarks}")
    try:
        tasks = mteb.get_benchmarks(names=args.benchmarks)
        print(f"✅ Loaded {len(tasks)} benchmark tasks")
        for task in tasks:
            print(f"   📋 {task.metadata.name}")
    except Exception as e:
        print(f"❌ Failed to load benchmarks: {e}")
        # Try to list available benchmarks
        try:
            all_benchmarks = mteb.get_benchmarks()
            vidore_benchmarks = [b for b in all_benchmarks if 'vidore' in b.metadata.name.lower()]
            if vidore_benchmarks:
                print(f"Available ViDoRe benchmarks: {[b.metadata.name for b in vidore_benchmarks]}")
            else:
                print("No ViDoRe benchmarks found. Available benchmarks:")
                for b in all_benchmarks[:10]:  # Show first 10
                    print(f"   - {b.metadata.name}")
        except:
            pass
        sys.exit(1)
    
    # Create evaluator
    evaluator = mteb.MTEB(tasks=tasks)
    
    # Run evaluation
    print("🔄 Starting evaluation...")
    print("⏱️  This may take several hours depending on the benchmark size...")
    
    try:
        results = evaluator.run(
            model,
            output_folder=args.output_dir,
            batch_size=args.batch_size,
            verbosity=2,
        )
        
        print("\n" + "🎉" + "="*58 + "🎉")
        print("🎉 EVALUATION COMPLETED SUCCESSFULLY! 🎉")
        print("🎉" + "="*58 + "🎉")
        print(f"📁 Results saved to: {args.output_dir}")
        
        # Print summary
        if results:
            print("\n📊 RESULTS SUMMARY:")
            print("-" * 40)
            for task_name, task_results in results.items():
                if isinstance(task_results, dict):
                    main_score = task_results.get('main_score', 'N/A')
                    print(f"📈 {task_name}: {main_score}")
                else:
                    print(f"📈 {task_name}: {task_results}")
            print("-" * 40)
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
