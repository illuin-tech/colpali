#!/usr/bin/env python3
"""
Quick test to verify MTEB integration with ColIntern3.5 works before running full evaluation.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_mteb_integration():
    print("Testing MTEB integration with ColIntern3.5...")
    
    try:
        # Import required modules
        import mteb
        from mteb.model_meta import ModelMeta
        from mteb_wrappers.colintern3_5_models import ColIntern3_5Wrapper
        from functools import partial
        
        print("✅ All imports successful")
        
        # Create model metadata
        custom_model_meta = ModelMeta(
            loader=partial(
                ColIntern3_5Wrapper,
                model_name="experiments/colintern3_5-1B-lora/checkpoint-1847"
            ),
            name="local/colintern3_5-test",
            modalities=["image", "text"],
            framework=["ColPali"],  # Must be a list
            similarity_fn_name="max_sim",
            languages=["eng-Latn"],
            revision="checkpoint-1847",
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
            training_datasets={"DocVQA": ["train"], "InfoVQA": ["train"]},  # Required field
        )
        
        print("✅ Model metadata created")
        
        # Load model
        model = custom_model_meta.load_model()
        print("✅ Model loaded successfully")
        
        # Check if ViDoRe benchmarks are available
        try:
            tasks = mteb.get_benchmarks(names=["ViDoRe(v1)"])
            print(f"✅ ViDoRe(v1) benchmark found with {len(tasks)} tasks")
        except Exception as e:
            print(f"⚠️  ViDoRe(v1) not available: {e}")
            try:
                # Try to get available benchmarks
                all_benchmarks = mteb.get_benchmarks()
                vidore_benchmarks = [b for b in all_benchmarks if 'vidore' in b.name.lower()]
                print(f"Available ViDoRe-like benchmarks: {[b.name for b in vidore_benchmarks]}")
            except:
                print("Could not list available benchmarks")
        
        print("✅ MTEB integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mteb_integration()
    sys.exit(0 if success else 1)
