#!/usr/bin/env python3
"""
Verify that all target modules for LoRA training exist in the ColIntern3_5 model.
This script validates the target_modules configuration for InternVL3.5-1B-HF.
"""

import sys
from pathlib import Path

# Add the colpali project root to Python path
_THIS_FILE = Path(__file__).resolve()
_COLPALI_DIR = _THIS_FILE.parents[1]  # Go up one level from scripts/
if str(_COLPALI_DIR) not in sys.path:
    sys.path.insert(0, str(_COLPALI_DIR))

from colpali_engine.models import ColIntern3_5
import torch
import re


def verify_target_modules():
    """Verify all target modules exist in the model."""
    
    print("Loading ColIntern3_5 model...")
    model = ColIntern3_5.from_pretrained(
        'OpenGVLab/InternVL3_5-1B-HF',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Target modules from training script
    target_modules = [
        "language_model.layers.*.self_attn.q_proj",
        "language_model.layers.*.self_attn.k_proj", 
        "language_model.layers.*.self_attn.v_proj",
        "language_model.layers.*.self_attn.o_proj",
        "language_model.layers.*.mlp.gate_proj",
        "language_model.layers.*.mlp.up_proj",
        "language_model.layers.*.mlp.down_proj",
        "custom_text_proj",
    ]
    
    print("\n" + "="*80)
    print("LORA TARGET MODULE VERIFICATION FOR InternVL3.5-1B-HF")
    print("="*80)
    
    # Get all module names (not just parameters)
    all_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            all_module_names.add(name)
    
    # Get all parameter names
    all_param_names = set(name.replace('.weight', '').replace('.bias', '') 
                         for name, _ in model.named_parameters())
    
    total_matched_modules = 0
    
    for target in target_modules:
        print(f"\n📍 Target: {target}")
        
        if '*' in target:
            # Pattern matching for wildcard targets
            pattern = target.replace('*', r'(\d+)')
            regex = re.compile(pattern)
            
            # Find matches in module names
            module_matches = [name for name in all_module_names if regex.match(name)]
            param_matches = [name for name in all_param_names if regex.match(name)]
            
            matches = list(set(module_matches + param_matches))
            
            if matches:
                print(f"  ✅ Found {len(matches)} matches")
                total_matched_modules += len(matches)
                
                # Show layer distribution
                layers = set()
                for match in matches:
                    layer_match = re.search(r'layers\.(\d+)\.', match)
                    if layer_match:
                        layers.add(int(layer_match.group(1)))
                
                if layers:
                    layer_range = f"layers {min(layers)}-{max(layers)}"
                    print(f"     Covers {layer_range} ({len(layers)} total layers)")
                
                # Show a few examples
                for i, match in enumerate(sorted(matches)[:3]):
                    print(f"     Example: {match}")
                if len(matches) > 3:
                    print(f"     ... and {len(matches) - 3} more")
            else:
                print("  ❌ No matches found")
                
        else:
            # Direct module name
            if target in all_module_names or target in all_param_names:
                print("  ✅ Found")
                total_matched_modules += 1
                
                # Check if it's a Linear layer
                if hasattr(model, target):
                    module = getattr(model, target)
                    if isinstance(module, torch.nn.Linear):
                        print(f"     Type: Linear({module.in_features} → {module.out_features})")
                    else:
                        print(f"     Type: {type(module).__name__}")
            else:
                print("  ❌ Not found")
    
    print(f"\n" + "="*80)
    print(f"SUMMARY")
    print("="*80)
    print(f"✅ Total modules that will receive LoRA adapters: {total_matched_modules}")
    
    # Architecture summary
    layer_count = len([name for name in all_module_names if 'language_model.layers.' in name and 'self_attn.q_proj' in name])
    print(f"📊 Language model layers: {layer_count}")
    print(f"📊 Attention projections per layer: 4 (q_proj, k_proj, v_proj, o_proj)")
    print(f"📊 MLP projections per layer: 3 (gate_proj, up_proj, down_proj)")
    print(f"📊 Custom projection: 1 (custom_text_proj)")
    print(f"📊 Expected total modules: {layer_count * 7 + 1} = {layer_count * 7 + 1}")
    
    # Verify the calculation
    expected_total = layer_count * 7 + 1  # 7 per layer + custom_text_proj
    if total_matched_modules == expected_total:
        print(f"✅ Module count matches expectation!")
    else:
        print(f"⚠️  Module count mismatch! Expected {expected_total}, found {total_matched_modules}")
    
    print(f"\n📋 RECOMMENDED PEFT CONFIG:")
    print("="*80)
    print("""
target_modules=[
    "language_model.layers.*.self_attn.q_proj",
    "language_model.layers.*.self_attn.k_proj", 
    "language_model.layers.*.self_attn.v_proj",
    "language_model.layers.*.self_attn.o_proj",
    "language_model.layers.*.mlp.gate_proj",
    "language_model.layers.*.mlp.up_proj",
    "language_model.layers.*.mlp.down_proj",
    "custom_text_proj",
]
""")
    
    print(f"\n🔍 WHAT GETS EXCLUDED (correctly):")
    print("="*80)
    print("❌ Vision tower (vision_tower.*) - we don't want to train vision encoder")
    print("❌ Multi-modal projector (multi_modal_projector.*) - connects vision to language")
    print("❌ Language model embeddings - typically frozen")
    print("❌ Layer norms and rotary embeddings - typically frozen")
    
    # Check what we're excluding
    excluded_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            is_targeted = False
            for target in target_modules:
                if '*' in target:
                    pattern = target.replace('*', r'\d+')
                    if re.match(pattern, name):
                        is_targeted = True
                        break
                else:
                    if name == target:
                        is_targeted = True
                        break
            
            if not is_targeted:
                excluded_modules.append(name)
    
    if excluded_modules:
        print(f"\n🚫 EXCLUDED LINEAR MODULES ({len(excluded_modules)} modules):")
        print("="*80)
        for module in excluded_modules:
            print(f"   {module}")
    
    print(f"\n✨ VERIFICATION COMPLETE!")
    return total_matched_modules == expected_total


if __name__ == "__main__":
    success = verify_target_modules()
    exit(0 if success else 1)
