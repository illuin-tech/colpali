"""
ColIntern3_5 Target Modules Verification Tests

This module contains tests for verifying that all LoRA target modules
are correctly identified and available in the ColIntern3.5 model.
"""

import logging
import re
from typing import Dict, List, Set

import pytest
import torch

from colpali_engine.models import ColIntern3_5
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

logger = logging.getLogger(__name__)


class TestColIntern3_5_TargetModulesVerification:  # noqa N801
    """Test class for comprehensive target modules verification."""

    @pytest.fixture(scope="class")
    def model(self) -> ColIntern3_5:
        """Load ColIntern3_5 model for target modules verification."""
        device = get_torch_device("auto")
        model = ColIntern3_5.from_pretrained(
            "OpenGVLab/InternVL3_5-1B-HF",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device,
        )
        yield model
        tear_down_torch()

    @pytest.fixture(scope="class")
    def target_modules(self) -> List[str]:
        """Define the target modules for LoRA training."""
        return [
            "language_model.layers.*.self_attn.q_proj",
            "language_model.layers.*.self_attn.k_proj", 
            "language_model.layers.*.self_attn.v_proj",
            "language_model.layers.*.self_attn.o_proj",
            "language_model.layers.*.mlp.gate_proj",
            "language_model.layers.*.mlp.up_proj",
            "language_model.layers.*.mlp.down_proj",
            "custom_text_proj",
        ]

    def test_target_modules_exist(self, model: ColIntern3_5, target_modules: List[str]):
        """Test that all target modules exist in the model."""
        # Get all module names (not just parameters)
        all_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                all_module_names.add(name)
        
        # Get all parameter names
        all_param_names = set(
            name.replace('.weight', '').replace('.bias', '') 
            for name, _ in model.named_parameters()
        )
        
        total_matched_modules = 0
        
        for target in target_modules:
            logger.info(f"Verifying target: {target}")
            
            if '*' in target:
                # Pattern matching for wildcard targets
                pattern = target.replace('*', r'(\d+)')
                regex = re.compile(pattern)
                
                # Find matches in module names
                module_matches = [name for name in all_module_names if regex.match(name)]
                param_matches = [name for name in all_param_names if regex.match(name)]
                
                matches = list(set(module_matches + param_matches))
                
                assert matches, f"No matches found for target pattern: {target}"
                total_matched_modules += len(matches)
                logger.info(f"Found {len(matches)} matches for {target}")
                
            else:
                # Direct module name
                assert (target in all_module_names or target in all_param_names), \
                    f"Target module not found: {target}"
                total_matched_modules += 1
                logger.info(f"Found direct target: {target}")
        
        # Verify expected total count
        layer_count = len([
            name for name in all_module_names 
            if 'language_model.layers.' in name and 'self_attn.q_proj' in name
        ])
        expected_total = layer_count * 7 + 1  # 7 per layer + custom_text_proj
        
        assert total_matched_modules == expected_total, \
            f"Module count mismatch! Expected {expected_total}, found {total_matched_modules}"

    def test_layer_count_verification(self, model: ColIntern3_5):
        """Test that we have the expected number of language model layers."""
        layer_modules = []
        for name, module in model.named_modules():
            if 'language_model.layers.' in name and 'self_attn.q_proj' in name:
                layer_modules.append(name)
        
        # InternVL3.5-1B should have 28 layers
        expected_layers = 28
        actual_layers = len(layer_modules)
        
        assert actual_layers == expected_layers, \
            f"Expected {expected_layers} language model layers, found {actual_layers}"
        
        logger.info(f"Verified {actual_layers} language model layers")

    def test_attention_projections_complete(self, model: ColIntern3_5):
        """Test that each layer has all attention projections."""
        attention_projections = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        
        # Get all layers
        layers = set()
        for name, module in model.named_modules():
            if 'language_model.layers.' in name and 'self_attn' in name:
                layer_match = re.search(r'layers\.(\d+)\.', name)
                if layer_match:
                    layers.add(int(layer_match.group(1)))
        
        # Check each layer has all attention projections
        for layer_num in layers:
            for proj in attention_projections:
                target_name = f"language_model.layers.{layer_num}.self_attn.{proj}"
                assert any(target_name in name for name, _ in model.named_modules()), \
                    f"Missing {proj} in layer {layer_num}"
        
        total_attention_modules = len(layers) * len(attention_projections)
        logger.info(f"Verified {total_attention_modules} attention projection modules")

    def test_mlp_projections_complete(self, model: ColIntern3_5):
        """Test that each layer has all MLP projections."""
        mlp_projections = ['gate_proj', 'up_proj', 'down_proj']
        
        # Get all layers
        layers = set()
        for name, module in model.named_modules():
            if 'language_model.layers.' in name and 'mlp' in name:
                layer_match = re.search(r'layers\.(\d+)\.', name)
                if layer_match:
                    layers.add(int(layer_match.group(1)))
        
        # Check each layer has all MLP projections
        for layer_num in layers:
            for proj in mlp_projections:
                target_name = f"language_model.layers.{layer_num}.mlp.{proj}"
                assert any(target_name in name for name, _ in model.named_modules()), \
                    f"Missing {proj} in layer {layer_num}"
        
        total_mlp_modules = len(layers) * len(mlp_projections)
        logger.info(f"Verified {total_mlp_modules} MLP projection modules")

    def test_custom_text_proj_exists(self, model: ColIntern3_5):
        """Test that custom_text_proj exists and has correct dimensions."""
        assert hasattr(model, 'custom_text_proj'), "Model missing custom_text_proj"
        
        custom_proj = model.custom_text_proj
        assert isinstance(custom_proj, torch.nn.Linear), \
            f"custom_text_proj should be Linear, got {type(custom_proj)}"
        
        # Check dimensions - should be 1024 -> 128
        assert custom_proj.in_features == 1024, \
            f"custom_text_proj input dim should be 1024, got {custom_proj.in_features}"
        assert custom_proj.out_features == 128, \
            f"custom_text_proj output dim should be 128, got {custom_proj.out_features}"
        
        logger.info(f"Verified custom_text_proj: {custom_proj.in_features} → {custom_proj.out_features}")

    def test_excluded_modules_verification(self, model: ColIntern3_5, target_modules: List[str]):
        """Test that vision modules are correctly excluded from targets."""
        # Get all Linear modules
        all_linear_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                all_linear_modules.append(name)
        
        # Check which modules match our targets
        targeted_modules = set()
        for target in target_modules:
            if '*' in target:
                pattern = target.replace('*', r'\d+')
                regex = re.compile(pattern)
                for module_name in all_linear_modules:
                    if regex.match(module_name):
                        targeted_modules.add(module_name)
            else:
                if target in all_linear_modules:
                    targeted_modules.add(target)
        
        # Check that vision modules are excluded
        excluded_vision_modules = [
            name for name in all_linear_modules 
            if 'vision_tower' in name
        ]
        
        for vision_module in excluded_vision_modules:
            assert vision_module not in targeted_modules, \
                f"Vision module incorrectly targeted: {vision_module}"
        
        logger.info(f"Verified {len(excluded_vision_modules)} vision modules are excluded")

    def test_total_module_count_197(self, model: ColIntern3_5, target_modules: List[str]):
        """Test that exactly 197 modules will receive LoRA adapters."""
        # Get all module names
        all_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                all_module_names.add(name)
        
        # Get all parameter names
        all_param_names = set(
            name.replace('.weight', '').replace('.bias', '') 
            for name, _ in model.named_parameters()
        )
        
        total_matched_modules = 0
        
        for target in target_modules:
            if '*' in target:
                pattern = target.replace('*', r'(\d+)')
                regex = re.compile(pattern)
                
                module_matches = [name for name in all_module_names if regex.match(name)]
                param_matches = [name for name in all_param_names if regex.match(name)]
                
                matches = list(set(module_matches + param_matches))
                total_matched_modules += len(matches)
            else:
                if target in all_module_names or target in all_param_names:
                    total_matched_modules += 1
        
        # Should be exactly 197: 28 layers × 7 projections + 1 custom_text_proj
        expected_total = 197
        assert total_matched_modules == expected_total, \
            f"Expected exactly {expected_total} target modules, found {total_matched_modules}"
        
        logger.info(f"✅ Verified exactly {total_matched_modules} target modules for LoRA training")

    def test_peft_config_generation(self, target_modules: List[str]):
        """Test that the target modules can be used in PEFT config."""
        # This test ensures our target_modules list is properly formatted for PEFT
        assert isinstance(target_modules, list), "target_modules should be a list"
        assert len(target_modules) == 8, f"Expected 8 target patterns, got {len(target_modules)}"
        
        # Check that we have the right patterns
        expected_patterns = {
            "language_model.layers.*.self_attn.q_proj",
            "language_model.layers.*.self_attn.k_proj", 
            "language_model.layers.*.self_attn.v_proj",
            "language_model.layers.*.self_attn.o_proj",
            "language_model.layers.*.mlp.gate_proj",
            "language_model.layers.*.mlp.up_proj",
            "language_model.layers.*.mlp.down_proj",
            "custom_text_proj",
        }
        
        assert set(target_modules) == expected_patterns, \
            "target_modules don't match expected patterns"
        
        logger.info("✅ Verified PEFT config compatibility")
