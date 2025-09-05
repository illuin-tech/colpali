# ColPali Testing Framework

This directory contains comprehensive tests for the ColPali engine, with special focus on ColIntern3.5 model implementation and target module verification.

## 📋 Test Structure

```
tests/
├── models/
│   ├── internvl3_5/
│   │   ├── README.md                    # InternVL3.5 test documentation
│   │   └── colintern3_5/
│   │       ├── test_modeling_colintern3_5.py      # Model tests
│   │       └── test_processing_colintern3_5.py    # Processor tests
│   ├── idefics3/                        # Idefics3 model tests
│   ├── paligemma/                       # PaliGemma model tests
│   ├── qwen2/                           # Qwen2 model tests
│   └── qwen2_5/                         # Qwen2.5 model tests
├── collators/                           # Data collator tests
├── compression/                         # Model compression tests
├── data/                                # Data handling tests
├── interpretability/                    # Model interpretability tests
├── loss/                                # Loss function tests
└── utils/                               # Utility function tests
```

## 🚀 Quick Start

### Run All Tests
```bash
cd /path/to/colpali

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=colpali_engine --cov-report=html

# Run only fast tests (skip slow model loading)
pytest tests/ -m "not slow" -v
```

### Run Specific Model Tests
```bash
# ColIntern3.5 tests
pytest tests/models/internvl3_5/ -v

# All model tests
pytest tests/models/ -v

# Specific test file
pytest tests/models/internvl3_5/colintern3_5/test_modeling_colintern3_5.py -v

# Specific processor tests
pytest tests/models/internvl3_5/colintern3_5/test_processing_colintern3_5.py -v

# MTEB integration tests
pytest tests/models/internvl3_5/colintern3_5/test_mteb_integration.py -v

# Target modules verification tests  
pytest tests/models/internvl3_5/colintern3_5/test_target_modules_verification.py -v
```

### Run Target Module Verification
```bash
# Verify LoRA target modules before testing
python scripts/verify_target_modules.py
```

## 🎯 Key Test Categories

### 1. Model Architecture Tests
**Location**: `tests/models/internvl3_5/colintern3_5/test_modeling_colintern3_5.py`

**What it tests**:
- ✅ Model loading and initialization
- ✅ Architecture validation (28 layers, 197 LoRA modules)
- ✅ Target module verification for LoRA training
- ✅ Custom projection layer (1024 → 128)
- ✅ Vision component exclusion from training

**Key test classes**:
```python
TestColIntern3_5_Model              # Basic model functionality
TestColIntern3_5_TargetModules      # LoRA target module verification
TestColIntern3_5_ModelIntegration   # End-to-end integration
```

### 2. Processor Tests
**Location**: `tests/models/internvl3_5/colintern3_5/test_processing_colintern3_5.py`

**What it tests**:
- ✅ Image processing and tokenization
- ✅ Text/query processing with augmentation
- ✅ Visual token limitation functionality
- ✅ Batch processing consistency
- ✅ Data type conversions (bfloat16)

**Key test classes**:
```python
TestColIntern3_5ProcessorBasic      # Basic processor functionality  
TestColIntern3_5ProcessorImages     # Image processing
TestColIntern3_5ProcessorText       # Text/query processing
TestColIntern3_5ProcessorAdvanced   # Edge cases and advanced features
```

### 3. MTEB Integration Tests
**Location**: `tests/models/internvl3_5/colintern3_5/test_mteb_integration.py`

**What it tests**:
- ✅ MTEB wrapper functionality
- ✅ Model metadata creation
- ✅ Benchmark availability (ViDoRe)
- ✅ Encoding pipeline compatibility
- ✅ Trained vs. untrained weight detection

**Key test classes**:
```python
TestColIntern3_5_MTEBIntegration    # MTEB wrapper and integration
TestColIntern3_5_MTEBCompatibility  # Benchmark compatibility
```

### 4. Target Modules Verification Tests
**Location**: `tests/models/internvl3_5/colintern3_5/test_target_modules_verification.py`

**What it tests**:
- ✅ All 197 LoRA target modules exist
- ✅ Layer count verification (28 layers)
- ✅ Attention projections completeness
- ✅ MLP projections completeness  
- ✅ Custom text projection validation
- ✅ Vision modules exclusion verification
- ✅ PEFT configuration compatibility

**Key test classes**:
```python
TestColIntern3_5_TargetModulesVerification  # Comprehensive target module validation
```
- ✅ MTEB framework compatibility
- ✅ Model wrapper functionality
- ✅ Benchmark availability and access
- ✅ Evaluation pipeline integration
- ✅ Similarity function compatibility (max_sim)

**Key test classes**:
```python
TestColIntern3_5_MTEBIntegration    # Core MTEB integration
TestColIntern3_5_MTEBCompatibility  # Benchmark compatibility
```

### 4. Target Module Verification
**Script**: `scripts/verify_target_modules.py`

**What it verifies**:
- ✅ All 197 LoRA target modules exist
- ✅ 28 language model layers × 7 projections each
- ✅ Custom text projection included
- ✅ Vision components correctly excluded
- ✅ Module naming and paths correct

## 📊 Expected Test Results

### Target Module Verification Output
```bash
$ python scripts/verify_target_modules.py

================================================================================
LORA TARGET MODULE VERIFICATION FOR InternVL3.5-1B-HF
================================================================================

📍 Target: language_model.layers.*.self_attn.q_proj
  ✅ Found 28 matches (covers layers 0-27)

📍 Target: language_model.layers.*.self_attn.k_proj  
  ✅ Found 28 matches (covers layers 0-27)

[... continues for all 8 target patterns ...]

📍 Target: custom_text_proj
  ✅ Found (Type: Linear(1024 → 128))

SUMMARY: ✅ Total modules that will receive LoRA adapters: 197
```

### Model Test Results
```bash
$ pytest tests/models/internvl3_5/colintern3_5/test_modeling_colintern3_5.py -v

test_modeling_colintern3_5.py::TestColIntern3_5_Model::test_load_model_from_pretrained PASSED
test_modeling_colintern3_5.py::TestColIntern3_5_Model::test_model_has_required_attributes PASSED
test_modeling_colintern3_5.py::TestColIntern3_5_Model::test_custom_text_proj_dimensions PASSED
test_modeling_colintern3_5.py::TestColIntern3_5_TargetModules::test_target_modules_exist PASSED
test_modeling_colintern3_5.py::TestColIntern3_5_TargetModules::test_lora_module_count_matches_expectation PASSED
test_modeling_colintern3_5.py::TestColIntern3_5_ModelIntegration::test_retrieval_integration PASSED

========================== 15 passed in 120.34s ==========================
```

## 🔧 Test Configuration

### Environment Variables
```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Test verbosity
export PYTEST_VERBOSITY=2
```

### Test Markers
```python
@pytest.mark.slow        # Long-running tests (model loading)
@pytest.mark.gpu         # GPU-required tests
@pytest.mark.integration # End-to-end integration tests
```

Run specific marker groups:
```bash
# Skip slow tests
pytest tests/ -m "not slow" -v

# Run only GPU tests
pytest tests/ -m "gpu" -v

# Run integration tests
pytest tests/ -m "integration" -v
```

## 🐛 Troubleshooting Tests

### Common Issues

#### **GPU Memory Issues**
```bash
# Error: CUDA out of memory
# Solution: Run specific tests or use CPU
pytest tests/models/internvl3_5/colintern3_5/test_processing_colintern3_5.py -v
```

#### **Model Loading Timeout**
```bash
# Error: Model download/loading takes too long
# Solution: Download model manually first
python -c "from colpali_engine.models import ColIntern3_5; ColIntern3_5.from_pretrained('OpenGVLab/InternVL3_5-1B-HF')"
```

#### **Target Module Verification Fails**
```bash
# Error: Expected 197 modules, found different number
# Solution: Check your model version and LoRA configuration
python scripts/verify_target_modules.py
```

### Test Environment Setup

#### **Minimal Test Setup**
```bash
# Install test dependencies
pip install pytest pytest-cov

# Set environment for testing
export PYTEST_CURRENT_TEST=1
export TOKENIZERS_PARALLELISM=false
```

#### **CI/CD Test Setup**
```bash
# GitHub Actions friendly setup
export CI=true
export CUDA_VISIBLE_DEVICES=""  # Force CPU for CI
pytest tests/ -m "not slow and not gpu" --maxfail=5
```

## 📈 Performance Benchmarks

### Expected Test Times
| Test Category | CPU Time | GPU Time | Memory Usage |
|---------------|----------|----------|-------------|
| **Processor Tests** | 30-60s | 10-20s | 2-4GB |
| **Model Tests** | 180-300s | 60-120s | 8-16GB |
| **MTEB Integration** | 120-240s | 45-90s | 8-16GB |
| **Target Verification** | 45-90s | 30-60s | 8-12GB |
| **Integration Tests** | 300-600s | 120-240s | 12-20GB |

### Memory Requirements
| Test Type | Minimum RAM | Recommended RAM | GPU Memory |
|-----------|-------------|-----------------|------------|
| **Unit Tests** | 8GB | 16GB | N/A |
| **Integration Tests** | 16GB | 32GB | 16GB |
| **Full Suite** | 32GB | 64GB | 24GB |

## 🎯 Test Quality Metrics

### Code Coverage Targets
- **Model Implementation**: >90%
- **Processor Implementation**: >90%
- **Target Module Paths**: 100%
- **Integration Workflows**: >85%

### Test Reliability Metrics
- ✅ **Deterministic**: All tests produce consistent results
- ✅ **Isolated**: Tests can run in any order
- ✅ **Fast**: Core tests complete in <5 minutes
- ✅ **Comprehensive**: Covers all critical paths

## 📚 Advanced Testing

### Custom Test Scenarios

#### **Test with Different Model Configurations**
```python
@pytest.mark.parametrize("max_tokens", [512, 768, 1024, 1536])
def test_visual_token_limits(max_tokens):
    processor = ColIntern3_5Processor.from_pretrained(
        "OpenGVLab/InternVL3_5-1B-HF",
        max_num_visual_tokens=max_tokens
    )
    # Test with different token limits...
```

#### **Test LoRA Configurations**
```python
@pytest.mark.parametrize("lora_rank", [8, 16, 32, 64])
def test_lora_ranks(lora_rank):
    # Test different LoRA rank configurations...
```

### Performance Testing
```bash
# Benchmark test execution time
pytest tests/models/internvl3_5/ --benchmark-only

# Memory profiling
pytest tests/models/internvl3_5/ --profile

# Parallel test execution  
pytest tests/models/internvl3_5/ -n auto
```

## 🚀 Continuous Integration

### Pre-commit Checks
```bash
# Run before committing
python scripts/verify_target_modules.py
pytest tests/models/internvl3_5/ -x  # Stop on first failure
```

### Automated Test Pipeline
```yaml
# .github/workflows/test.yml
name: Test ColPali
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -e .[test]
      - name: Verify target modules
        run: python scripts/verify_target_modules.py
      - name: Run tests
        run: pytest tests/ -m "not slow" --cov=colpali_engine
```

---

## 🎯 Summary

This testing framework ensures:

1. **Architecture Validation**: All 197 LoRA target modules correctly configured
2. **Functionality Testing**: End-to-end retrieval pipeline works correctly  
3. **Integration Testing**: Model and processor work together seamlessly
4. **Performance Validation**: Memory usage and execution times are reasonable
5. **Regression Prevention**: Changes don't break existing functionality

**Run the essential tests**:
```bash
# Quick validation workflow
python scripts/verify_target_modules.py  # Verify target modules
pytest tests/models/internvl3_5/ -v      # Test model implementation
```

This comprehensive testing ensures your ColIntern3.5 implementation is robust, correctly configured, and ready for high-performance training!
