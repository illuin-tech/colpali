import logging
from typing import Generator, cast

import pytest
import torch
from datasets import load_dataset
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColIntern3_5, ColIntern3_5Processor
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def model_name() -> str:
    """Model name for InternVL3.5-1B-HF testing."""
    return "OpenGVLab/InternVL3_5-1B-HF"


@pytest.fixture(scope="module")
def model_without_mask(model_name: str) -> Generator[ColIntern3_5, None, None]:
    """Load ColIntern3_5 model for testing without masking."""
    device = get_torch_device("auto")
    logger.info(f"Device used: {device}")

    yield cast(
        ColIntern3_5,
        ColIntern3_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            trust_remote_code=True,
        ).eval(),
    )
    tear_down_torch()


@pytest.fixture(scope="module")
def model_with_mask(model_name: str) -> Generator[ColIntern3_5, None, None]:
    """Load ColIntern3_5 model for testing with masking."""
    device = get_torch_device("auto")
    logger.info(f"Device used: {device}")

    yield cast(
        ColIntern3_5,
        ColIntern3_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            trust_remote_code=True,
        ).eval(),
    )
    tear_down_torch()


@pytest.fixture(scope="module")
def processor(model_name: str) -> Generator[ColIntern3_5Processor, None, None]:
    """Load ColIntern3_5 processor with visual token limitation."""
    yield cast(
        ColIntern3_5Processor, 
        ColIntern3_5Processor.from_pretrained(
            model_name, 
            max_num_visual_tokens=768
        )
    )


class TestColIntern3_5_Model:  # noqa N801
    """Test basic ColIntern3_5 model functionality and architecture."""
    
    @pytest.mark.slow
    def test_load_model_from_pretrained(self, model_without_mask: ColIntern3_5):
        """Test that the model loads correctly from HuggingFace hub."""
        assert isinstance(model_without_mask, ColIntern3_5)
        
    def test_model_has_required_attributes(self, model_without_mask: ColIntern3_5):
        """Test that the model has all required attributes for LoRA training."""
        assert hasattr(model_without_mask, 'language_model')
        assert hasattr(model_without_mask, 'custom_text_proj')
        
    def test_model_dtype(self, model_without_mask: ColIntern3_5):
        """Test that the model uses the correct dtype (bfloat16)."""
        assert model_without_mask.custom_text_proj.weight.dtype == torch.bfloat16
        
    def test_custom_text_proj_dimensions(self, model_without_mask: ColIntern3_5):
        """Test that custom_text_proj has expected dimensions (1024 → 128)."""
        custom_proj = model_without_mask.custom_text_proj
        assert isinstance(custom_proj, torch.nn.Linear)
        assert custom_proj.in_features == 1024
        assert custom_proj.out_features == 128
        
    def test_language_model_structure(self, model_without_mask: ColIntern3_5):
        """Test that language model has expected layer structure."""
        language_model = model_without_mask.language_model
        assert hasattr(language_model, 'layers')
        
        # Should have 28 layers for InternVL3.5-1B
        assert len(language_model.layers) == 28
        
        # Each layer should have attention and MLP components
        first_layer = language_model.layers[0]
        assert hasattr(first_layer, 'self_attn')
        assert hasattr(first_layer, 'mlp')
        
        # Check attention projections
        self_attn = first_layer.self_attn
        assert hasattr(self_attn, 'q_proj')
        assert hasattr(self_attn, 'k_proj')
        assert hasattr(self_attn, 'v_proj')
        assert hasattr(self_attn, 'o_proj')
        
        # Check MLP projections
        mlp = first_layer.mlp
        assert hasattr(mlp, 'gate_proj')
        assert hasattr(mlp, 'up_proj')
        assert hasattr(mlp, 'down_proj')


class TestColIntern3_5_TargetModules:  # noqa N801
    """Test LoRA target module verification for InternVL3.5-1B."""
    
    def test_target_modules_exist(self, model_without_mask: ColIntern3_5):
        """Test that all expected LoRA target modules exist in the model."""
        import re
        
        # Target modules from training configuration
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
        
        # Get all module names
        all_module_names = set()
        for name, module in model_without_mask.named_modules():
            if isinstance(module, torch.nn.Linear):
                all_module_names.add(name)
        
        total_matched_modules = 0
        
        for target in target_modules:
            if '*' in target:
                # Pattern matching for wildcard targets
                pattern = target.replace('*', r'(\d+)')
                regex = re.compile(pattern)
                matches = [name for name in all_module_names if regex.match(name)]
                
                if 'self_attn' in target or 'mlp' in target:
                    # Should match 28 layers
                    assert len(matches) == 28, f"Expected 28 matches for {target}, got {len(matches)}"
                
                total_matched_modules += len(matches)
            else:
                # Direct module name (custom_text_proj)
                assert target in all_module_names, f"Target module {target} not found"
                total_matched_modules += 1
        
        # Should have exactly 197 modules (28 layers × 7 projections + 1 custom)
        expected_total = 28 * 7 + 1  # 196 + 1
        assert total_matched_modules == expected_total, \
            f"Expected {expected_total} target modules, found {total_matched_modules}"
    
    def test_attention_projections_exist(self, model_without_mask: ColIntern3_5):
        """Test that all attention projections exist across all layers."""
        language_model = model_without_mask.language_model
        
        for layer_idx in range(28):  # 28 layers
            layer = language_model.layers[layer_idx]
            self_attn = layer.self_attn
            
            # Check all attention projections
            assert hasattr(self_attn, 'q_proj'), f"Layer {layer_idx} missing q_proj"
            assert hasattr(self_attn, 'k_proj'), f"Layer {layer_idx} missing k_proj"
            assert hasattr(self_attn, 'v_proj'), f"Layer {layer_idx} missing v_proj"
            assert hasattr(self_attn, 'o_proj'), f"Layer {layer_idx} missing o_proj"
            
            # Check they are Linear modules
            assert isinstance(self_attn.q_proj, torch.nn.Linear)
            assert isinstance(self_attn.k_proj, torch.nn.Linear)
            assert isinstance(self_attn.v_proj, torch.nn.Linear)
            assert isinstance(self_attn.o_proj, torch.nn.Linear)
    
    def test_mlp_projections_exist(self, model_without_mask: ColIntern3_5):
        """Test that all MLP projections exist across all layers."""
        language_model = model_without_mask.language_model
        
        for layer_idx in range(28):  # 28 layers
            layer = language_model.layers[layer_idx]
            mlp = layer.mlp
            
            # Check all MLP projections
            assert hasattr(mlp, 'gate_proj'), f"Layer {layer_idx} missing gate_proj"
            assert hasattr(mlp, 'up_proj'), f"Layer {layer_idx} missing up_proj"
            assert hasattr(mlp, 'down_proj'), f"Layer {layer_idx} missing down_proj"
            
            # Check they are Linear modules
            assert isinstance(mlp.gate_proj, torch.nn.Linear)
            assert isinstance(mlp.up_proj, torch.nn.Linear)
            assert isinstance(mlp.down_proj, torch.nn.Linear)
    
    def test_vision_modules_excluded(self, model_without_mask: ColIntern3_5):
        """Test that vision modules are correctly excluded from LoRA targets."""
        # Vision modules should exist but not be in target_modules
        assert hasattr(model_without_mask, 'vision_tower')
        assert hasattr(model_without_mask, 'multi_modal_projector')
        
        # Count vision tower modules (should be excluded)
        vision_modules = []
        for name, module in model_without_mask.vision_tower.named_modules():
            if isinstance(module, torch.nn.Linear):
                vision_modules.append(f"vision_tower.{name}")
        
        # Should have many vision modules (expected ~144)
        assert len(vision_modules) > 100, f"Expected >100 vision modules, found {len(vision_modules)}"
        
        # Multi-modal projector modules (should be excluded)
        mm_modules = []
        for name, module in model_without_mask.multi_modal_projector.named_modules():
            if isinstance(module, torch.nn.Linear):
                mm_modules.append(f"multi_modal_projector.{name}")
        
        # Should have 2 multi-modal projector modules
        assert len(mm_modules) == 2, f"Expected 2 multi-modal modules, found {len(mm_modules)}"
    
    def test_lora_module_count_matches_expectation(self, model_without_mask: ColIntern3_5):
        """Test that the total LoRA module count matches our verification script."""
        # This should match the output of scripts/verify_target_modules.py
        
        # Count language model linear modules that are LoRA targets
        language_modules = 0
        for name, module in model_without_mask.language_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check if it's a target module
                if ('self_attn' in name and any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])) or \
                   ('mlp' in name and any(proj in name for proj in ['gate_proj', 'up_proj', 'down_proj'])):
                    language_modules += 1
        
        # Should have 196 language model modules (28 layers × 7 projections)
        assert language_modules == 196, f"Expected 196 language modules, found {language_modules}"
        
        # Plus 1 custom_text_proj = 197 total
        total_lora_modules = language_modules + 1
        assert total_lora_modules == 197, f"Expected 197 total LoRA modules, found {total_lora_modules}"


class TestColIntern3_5_ModelIntegration:  # noqa N801
    """Test ColIntern3_5 integration and end-to-end functionality."""
    
    @pytest.mark.slow
    def test_forward_images_integration(
        self,
        model_without_mask: ColIntern3_5,
        processor: ColIntern3_5Processor,
    ):
        """Test that model can handle image processing without errors."""
        # Create a batch of dummy images with different sizes
        images = [
            Image.new("RGB", (224, 224), color="white"),
            Image.new("RGB", (448, 448), color="black"),
        ]

        # Process the image
        batch_images = processor.process_images(images).to(model_without_mask.device)

        # Forward pass
        with torch.no_grad():
            outputs = model_without_mask(**batch_images)

        # Assertions
        assert isinstance(outputs, torch.Tensor)
        assert outputs.dim() == 3
        batch_size, n_visual_tokens, emb_dim = outputs.shape
        assert batch_size == len(images)
        assert emb_dim == 128  # ColIntern3_5 embedding dimension

    @pytest.mark.slow
    def test_forward_queries_integration(
        self,
        model_without_mask: ColIntern3_5,
        processor: ColIntern3_5Processor,
    ):
        """Test query processing and forward pass."""
        queries = [
            "What is shown in this image?",
            "Describe the content of this document.",
        ]

        # Process the queries
        batch_queries = processor.process_queries(queries).to(model_without_mask.device)

        # Forward pass
        with torch.no_grad():
            outputs = model_without_mask(**batch_queries)

        # Assertions
        assert isinstance(outputs, torch.Tensor)
        assert outputs.dim() == 3
        batch_size, n_query_tokens, emb_dim = outputs.shape
        assert batch_size == len(queries)
        assert emb_dim == 128  # ColIntern3_5 embedding dimension

    @pytest.mark.slow
    def test_retrieval_integration(
        self,
        model_without_mask: ColIntern3_5,
        processor: ColIntern3_5Processor,
    ):
        """Test retrieval capabilities."""
        # Load the test dataset
        ds = load_dataset("hf-internal-testing/document-visual-retrieval-test", split="test")

        # Preprocess the examples
        batch_images = processor.process_images(images=ds["image"]).to(model_without_mask.device)
        batch_queries = processor.process_queries(queries=ds["query"]).to(model_without_mask.device)

        # Run inference
        with torch.inference_mode():
            image_embeddings = model_without_mask(**batch_images)
            query_embeddings = model_without_mask(**batch_queries)

        # Compute retrieval scores
        scores = processor.score_multi_vector(
            qs=query_embeddings,
            ps=image_embeddings,
        )  # (len(qs), len(ps))

        assert scores.ndim == 2, f"Expected 2D tensor, got {scores.ndim}"
        assert scores.shape == (len(ds), len(ds)), f"Expected shape {(len(ds), len(ds))}, got {scores.shape}"

        # For an untrained model, we just check that the retrieval mechanism works
        # (produces valid scores) rather than expecting perfect accuracy
        assert torch.isfinite(scores).all(), "All scores should be finite"
        assert not torch.isnan(scores).any(), "No scores should be NaN"
        
        # Check that scores are reasonable (not all identical)
        assert scores.std() > 0, "Scores should have some variance"
        
        # Check that argmax produces valid indices
        max_indices = scores.argmax(dim=1)
        assert (max_indices >= 0).all() and (max_indices < len(ds)).all(), "Argmax indices should be valid"

    def test_output_consistency(
        self,
        model_without_mask: ColIntern3_5,
        processor: ColIntern3_5Processor,
    ):
        """Test that image and query outputs have consistent dimensionality."""
        # Create test inputs
        image = Image.new("RGB", (224, 224), color="white")
        query = "Test query"
        
        # Process inputs
        batch_image = processor.process_images([image]).to(model_without_mask.device)
        batch_query = processor.process_queries([query]).to(model_without_mask.device)
        
        # Forward pass
        with torch.no_grad():
            image_output = model_without_mask(**batch_image)
            query_output = model_without_mask(**batch_query)
        
        # Check consistency
        assert image_output.shape[0] == query_output.shape[0] == 1  # Same batch size
        assert image_output.shape[2] == query_output.shape[2] == 128  # Same embedding dimension

    def test_visual_token_limitation(
        self,
        model_without_mask: ColIntern3_5,
        processor: ColIntern3_5Processor,
    ):
        """Test that visual token limitation is working."""
        # Create a moderately large image 
        large_image = Image.new("RGB", (800, 800), color="white")
        
        # Process the image
        batch_image = processor.process_images([large_image]).to(model_without_mask.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model_without_mask(**batch_image)
        
        # Check that the number of tokens is reasonable
        _, n_tokens, _ = outputs.shape
        assert n_tokens < 2000, f"Too many visual tokens: {n_tokens}. Token limitation may not be working properly."

    def test_different_image_sizes(
        self,
        model_without_mask: ColIntern3_5,
        processor: ColIntern3_5Processor,
    ):
        """Test processing images of different sizes."""
        images = [
            Image.new("RGB", (224, 224), color="red"),
            Image.new("RGB", (448, 448), color="green"),
            Image.new("RGB", (300, 500), color="blue"),
            Image.new("RGB", (600, 400), color="yellow"),
        ]
        
        # Process images
        batch_images = processor.process_images(images).to(model_without_mask.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model_without_mask(**batch_images)
        
        # Assertions
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape[0] == len(images)
        assert outputs.shape[2] == 128  # Embedding dimension
        assert outputs.shape[-1] == 128
