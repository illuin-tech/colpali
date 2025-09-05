# ColIntern3.5 Testing Documentation

This directory contains comprehensive tests for the ColIntern3.5 model implementation and target module verification.

## 📋 Test Overview

The testing suite validates:
- ✅ **Model Architecture**: ColIntern3.5 model loading and structure
- ✅ **Target Modules**: LoRA target module verification (197 modules)
- ✅ **Processor Implementation**: Image and text processing pipelines
- ✅ **Integration Testing**: End-to-end retrieval functionality
- ✅ **Performance Validation**: Model inference and output consistency

## 🧪 Test Files

### Core Model Tests
- **`test_modeling_colintern3_5.py`** - Model architecture and inference tests
- **`test_processing_colintern3_5.py`** - Processor and input handling tests
- **`test_mteb_integration.py`** - MTEB integration and benchmark compatibility tests
- **`test_target_modules_verification.py`** - LoRA target modules verification tests

### Verification Scripts
- **`scripts/verify_target_modules.py`** - Target module validation

## 🚀 Running Tests

### Run All ColIntern3.5 Tests
```bash
cd /path/to/colpali

# Run all tests for ColIntern3.5
pytest tests/models/internvl3_5/colintern3_5/ -v

# Run with detailed output
pytest tests/models/internvl3_5/colintern3_5/ -v -s

# Run specific test file
pytest tests/models/internvl3_5/colintern3_5/test_modeling_colintern3_5.py -v

# Run MTEB integration tests
pytest tests/models/internvl3_5/colintern3_5/test_mteb_integration.py -v

# Run target modules verification tests
pytest tests/models/internvl3_5/colintern3_5/test_target_modules_verification.py -v
```

### Run Target Module Verification
```bash
# Verify target modules before testing/training
python scripts/verify_target_modules.py
```

### Run Tests with Coverage
```bash
# Generate test coverage report
pytest tests/models/internvl3_5/colintern3_5/ --cov=colpali_engine.models.internvl3_5 --cov-report=html
```

## 📊 Test Categories

### 1. Model Architecture Tests (`test_modeling_colintern3_5.py`)

#### **Basic Model Loading**
```python
def test_load_model_from_pretrained()
def test_model_has_required_attributes()
def test_model_dtype()
```

**What it validates:**
- ✅ Model loads from `OpenGVLab/InternVL3_5-1B-HF`
- ✅ Required attributes exist (`language_model`, `custom_text_proj`)
- ✅ Model uses `torch.bfloat16` dtype

#### **Integration Tests**
```python
def test_forward_images_integration()
def test_forward_queries_integration()
def test_retrieval_integration()
```

**What it validates:**
- ✅ Image processing and forward pass
- ✅ Query processing and forward pass
- ✅ End-to-end retrieval scoring
- ✅ Output tensor shapes and dimensions

#### **Output Consistency Tests**
```python
def test_output_consistency()
def test_visual_token_limitation()
def test_different_image_sizes()
```

**What it validates:**
- ✅ Image and query outputs have consistent embedding dimensions (128D)
- ✅ Visual token limitation prevents memory issues
- ✅ Different image sizes are handled correctly

### 2. Processor Tests (`test_processing_colintern3_5.py`)

#### **Basic Processor Tests**
```python
def test_load_processor_from_pretrained()
def test_processor_has_required_attributes()
def test_tokenizer_padding_side()
```

**What it validates:**
- ✅ Processor loads with correct configuration
- ✅ Required attributes exist (`tokenizer`, `image_processor`)
- ✅ Tokenizer padding is set to "right"

#### **Image Processing Tests**
```python
def test_process_images()
def test_process_images_multiple()
def test_process_images_dtype_conversion()
```

**What it validates:**
- ✅ Single image processing produces correct tensors
- ✅ Multiple image batching works correctly
- ✅ Pixel values converted to `torch.bfloat16`

#### **Text Processing Tests**
```python
def test_process_texts()
def test_process_queries()
def test_process_queries_vs_texts_equivalence()
```

**What it validates:**
- ✅ Text tokenization and encoding
- ✅ Query processing with augmentation tokens
- ✅ Equivalence between text and query processing

### 3. MTEB Integration Tests (`test_mteb_integration.py`)

#### **MTEB Compatibility Tests**
```python
def test_mteb_imports()
def test_model_wrapper_loading()
def test_model_metadata_creation()
```

**What it validates:**
- ✅ MTEB dependencies can be imported
- ✅ ColIntern3_5Wrapper loads models correctly
- ✅ ModelMeta can be created with proper metadata

#### **Benchmark Integration Tests**
```python
def test_vidore_benchmark_availability()
def test_evaluation_pipeline_compatibility()
def test_similarity_function_compatibility()
```

**What it validates:**
- ✅ ViDoRe benchmarks are accessible
- ✅ Model works with MTEB evaluation pipeline
- ✅ max_sim similarity function produces valid scores
- ✅ Trained weight detection works correctly

## 🎯 Target Module Verification Tests

The verification script validates your LoRA configuration:

### Expected Test Results
```bash
$ python scripts/verify_target_modules.py

================================================================================
LORA TARGET MODULE VERIFICATION FOR InternVL3.5-1B-HF
================================================================================

📍 Target: language_model.layers.*.self_attn.q_proj
  ✅ Found 28 matches

📍 Target: language_model.layers.*.self_attn.k_proj
  ✅ Found 28 matches

📍 Target: language_model.layers.*.self_attn.v_proj
  ✅ Found 28 matches

📍 Target: language_model.layers.*.self_attn.o_proj
  ✅ Found 28 matches

📍 Target: language_model.layers.*.mlp.gate_proj
  ✅ Found 28 matches

📍 Target: language_model.layers.*.mlp.up_proj
  ✅ Found 28 matches

📍 Target: language_model.layers.*.mlp.down_proj
  ✅ Found 28 matches

📍 Target: custom_text_proj
  ✅ Found (Type: Linear(1024 → 128))

SUMMARY: ✅ Total modules that will receive LoRA adapters: 197
```

### Target Module Test Breakdown

| Test Category | Modules Tested | Expected Result |
|---------------|----------------|-----------------|
| **Attention Projections** | 112 (28 layers × 4) | ✅ All found |
| **MLP Projections** | 84 (28 layers × 3) | ✅ All found |
| **Custom Projection** | 1 | ✅ Found (1024→128) |
| **Vision Components** | 146 | ❌ Correctly excluded |

## 🐛 Test Troubleshooting

### Common Test Issues

#### **Issue: Model Loading Failures**
```bash
# Error: CUDA out of memory
pytest tests/models/internvl3_5/colintern3_5/ --tb=short

# Solution: Run on CPU or reduce test scope
pytest tests/models/internvl3_5/colintern3_5/test_processing_colintern3_5.py -v
```

#### **Issue: Target Module Verification Fails**
```bash
# Check if target modules are correctly configured
python scripts/verify_target_modules.py

# Expected: 197 total modules
# If different: Check your LoRA configuration
```

#### **Issue: Flash Attention Not Available**
```python
# Tests automatically fall back to standard attention
# No action needed - tests will pass with standard attention
```

### Test Environment Setup

#### **Minimal Test Environment**
```bash
# Install test dependencies
pip install pytest pytest-cov

# Set test environment variables
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Reduce memory fragmentation
```

#### **Test with Different Hardware**
```bash
# CPU-only testing
pytest tests/models/internvl3_5/colintern3_5/ --device=cpu

# GPU testing with memory monitoring
watch -n 1 nvidia-smi &
pytest tests/models/internvl3_5/colintern3_5/ -v
```

## 📈 Expected Test Performance

### Test Execution Times
| Test Category | Expected Time | GPU Memory |
|---------------|---------------|------------|
| **Model Loading** | 30-60 seconds | 8-12GB |
| **Processor Tests** | 5-10 seconds | 2-4GB |
| **Integration Tests** | 60-120 seconds | 12-16GB |
| **MTEB Integration** | 45-90 seconds | 8-16GB |
| **Target Verification** | 30-45 seconds | 8-12GB |

### Performance Benchmarks
```python
# Expected output shapes for integration tests
image_embeddings.shape  # torch.Size([batch_size, n_visual_tokens, 128])
query_embeddings.shape  # torch.Size([batch_size, n_query_tokens, 128])
scores.shape           # torch.Size([n_queries, n_images])

# Expected value ranges
assert 0.0 <= scores.max() <= 1.0  # Normalized scores
assert scores.std() > 0.01         # Meaningful variance
```

## 🔧 Advanced Testing

### Custom Test Configurations

#### **Test with Different Model Sizes**
```python
# Test with different max_num_visual_tokens
@pytest.mark.parametrize("max_tokens", [512, 768, 1024, 1536])
def test_visual_token_limits(max_tokens):
    processor = ColIntern3_5Processor.from_pretrained(
        "OpenGVLab/InternVL3_5-1B-HF",
        max_num_visual_tokens=max_tokens
    )
    # Test processing...
```

#### **Test with Different Batch Sizes**
```python
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_batch_processing(batch_size):
    images = [Image.new("RGB", (224, 224)) for _ in range(batch_size)]
    # Test batch processing...
```

### Integration with Training Tests

#### **Test Trained Model Loading**
```python
def test_trained_model_loading():
    """Test loading a trained model with PEFT adapters."""
    from mteb_wrappers.colintern3_5_models import ColIntern3_5Wrapper
    
    # Load trained model
    model = ColIntern3_5Wrapper('experiments/colintern3_5-trained')
    
    # Check that custom_text_proj has trained weights
    custom_proj = model.mdl.custom_text_proj
    weight_std = custom_proj.weight.std().item()
    
    # Trained weights should have different distribution than random
    assert weight_std != pytest.approx(0.02, abs=0.005), "Weights appear untrained"
```

## 📚 Test Data

### Test Datasets Used
```python
# Internal test dataset for retrieval testing
ds = load_dataset("hf-internal-testing/document-visual-retrieval-test", split="test")

# Custom test images
test_images = [
    Image.new("RGB", (224, 224), color="white"),
    Image.new("RGB", (448, 448), color="black"),
    Image.new("RGB", (300, 500), color="blue"),
]

# Test queries
test_queries = [
    "What is shown in this image?",
    "Describe the content of this document.",
    "Find relevant information.",
]
```

## 🎯 Test Quality Metrics

### Code Coverage Targets
- **Model Implementation**: >90% coverage
- **Processor Implementation**: >90% coverage
- **Target Module Verification**: 100% coverage
- **Integration Paths**: >80% coverage

### Test Reliability
- ✅ **Deterministic**: Tests produce consistent results
- ✅ **Isolated**: Each test can run independently
- ✅ **Fast**: Core tests complete in <5 minutes
- ✅ **Comprehensive**: Covers all critical functionality

---

## 🚀 Quick Test Commands

```bash
# Essential test workflow
cd /path/to/colpali

# 1. Verify target modules
python scripts/verify_target_modules.py

# 2. Run core tests
pytest tests/models/internvl3_5/colintern3_5/ -v

# 3. Run MTEB integration tests
pytest tests/models/internvl3_5/colintern3_5/test_mteb_integration.py -v

# 4. Run with coverage
pytest tests/models/internvl3_5/colintern3_5/ --cov=colpali_engine.models.internvl3_5

# 5. Test specific functionality
pytest tests/models/internvl3_5/colintern3_5/test_modeling_colintern3_5.py::TestColIntern3_5_ModelIntegration::test_retrieval_integration -v
```

This testing framework ensures your ColIntern3.5 implementation is robust, properly configured, and ready for high-performance training and evaluation!
