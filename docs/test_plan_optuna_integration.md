# Test Plan: ModelBatch-Optuna Integration

## Overview

This test plan outlines comprehensive testing for the ModelBatch-Optuna integration, covering constraint systems, batching logic, and end-to-end workflows.

## Test Categories

### 1. Unit Tests

#### Constraint System Tests
- **ConstraintSpec Tests**
  - Basic constraint specification creation
  - Constraint key generation for grouping
  - Parameter validation and overlap detection
  - Complex parameter nesting support

#### Batching Logic Tests
- **BatchGroup Tests**
  - Group creation and initialization
  - Trial addition and management
  - Batch readiness determination
  - Variable configuration extraction
  - Memory efficiency during model creation

- **TrialBatcher Tests**
  - Trial grouping by constraints
  - Ready batch identification
  - Batch status reporting
  - Constraint key generation consistency

#### Integration Tests
- **ModelBatchStudy Tests**
  - Study creation and initialization
  - Parameter suggestion (NotImplementedError)
  - Model creation for batches
  - Optimization summary generation

### 2. Integration Tests

#### End-to-End Workflows
- **Basic Optimization Workflow**
  - Complete optimization cycle with constraints
  - Trial lifecycle management
  - Batch execution and results
  - Progress tracking and reporting

- **Advanced Constraint Scenarios**
  - Multiple constraint groups
  - Progressive batch filling
  - Timeout-based batch starting
  - Memory-constrained scenarios

#### Error Handling
- **Invalid Parameters**
  - Overlapping fixed/variable parameters
  - Missing required parameters
  - Invalid constraint specifications

- **Batch Failures**
  - Handling of failed trials within batches
  - Error propagation and recovery
  - Partial batch completion

### 3. Performance Tests

#### Batching Efficiency
- **Batch Utilization**
  - Measure batch fill rates vs. optimal
  - GPU utilization during batch training
  - Memory usage patterns

- **Scaling Tests**
  - Performance with increasing batch sizes
  - Constraint grouping efficiency
  - Trial throughput measurement

#### Memory Management
- **Model Creation Overhead**
  - Memory usage during model instantiation
  - Garbage collection efficiency
  - Large model handling

### 4. HuggingFace Integration Tests

#### Model Creation
- **HFModelFactory Tests**
  - Model creation with constraints
  - Configuration validation
  - Tokenizer compatibility

#### Training Integration
- **ModelBatchHFTrainer Tests**
  - Training loop compatibility
  - Dataset preprocessing
  - Evaluation metrics
  - Checkpoint management

## Test Data and Fixtures

### Synthetic Data
- **Regression Dataset**
  - 1000 samples, 10 features
  - Random targets for loss computation
  - Consistent across test runs

### Model Architectures
- **SimpleMLP**
  - Configurable hidden size, dropout
  - Fast training for testing
  - Clear parameter structure

- **HuggingFace Models**
  - DistilBERT (small transformer)
  - Configurable parameters
  - Real-world complexity

### Constraint Scenarios
- **Basic Constraints**
  - Fixed architecture parameters
  - Variable optimizer parameters
  - Batch-aware parameters

- **Complex Constraints**
  - Nested parameter structures
  - Multiple constraint dimensions
  - Cross-parameter dependencies

## Test Execution Plan

### Phase 1: Core Functionality
1. Run constraint system tests
2. Test batching logic independently
3. Verify ModelBatchStudy basic functionality
4. Test error handling and edge cases

### Phase 2: Integration Workflows
1. Run end-to-end optimization demos
2. Test progressive batch filling
3. Verify constraint enforcement
4. Test memory efficiency

### Phase 3: Performance Validation
1. Run scaling benchmarks
2. Measure batch utilization
3. Test memory constraints
4. Validate performance improvements

### Phase 4: HuggingFace Integration
1. Test HF model creation
2. Run HF training workflows
3. Verify dataset compatibility
4. Test checkpoint management

## Test Commands

### Run All Tests
```bash
# Install test dependencies
pip install optuna transformers datasets

# Run basic tests
python -m pytest tests/test_optuna_integration.py -v

# Run specific test categories
python -m pytest tests/test_optuna_integration.py::TestConstraintSpec -v
python -m pytest tests/test_optuna_integration.py::TestBatchGroup -v
python -m pytest tests/test_optuna_integration.py::TestTrialBatcher -v

# Run integration tests
python -m pytest tests/test_optuna_integration.py::TestIntegration -v
```

### Performance Tests
```bash
# Run performance benchmarks
python examples/optuna_integration_demo.py

# Run with profiling
python -m cProfile -o profile.out examples/optuna_integration_demo.py
```

### HuggingFace Tests
```bash
# Run HF integration tests (if available)
python -m pytest tests/test_huggingface_integration.py -v
```

## Expected Outcomes

### Success Criteria
- **All unit tests pass** with >90% coverage
- **Integration tests complete** without errors
- **Performance benchmarks** show expected speedups
- **Memory usage** remains within acceptable bounds
- **Constraint enforcement** works correctly

### Performance Targets
- **Batch utilization** >80% for large studies
- **Trial throughput** >100 trials/hour for simple models
- **Memory overhead** <10% compared to individual training
- **Constraint grouping** <1ms per trial

## Test Validation

### Validation Checklist
- [ ] All unit tests pass
- [ ] Integration tests complete successfully
- [ ] Performance benchmarks meet targets
- [ ] Memory usage is efficient
- [ ] Error handling is robust
- [ ] Documentation is accurate
- [ ] Examples run without modification

### Benchmark Comparison
- [ ] Compare against sequential Optuna studies
- [ ] Measure speedup vs. individual training
- [ ] Verify constraint enforcement accuracy
- [ ] Validate batching efficiency

## Continuous Integration

### CI Pipeline
```yaml
# .github/workflows/tests.yml
name: ModelBatch-Optuna Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install optuna transformers datasets pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/test_optuna_integration.py -v --cov=modelbatch
    
    - name: Run integration tests
      run: |
        python examples/optuna_integration_demo.py
```

## Troubleshooting Guide

### Common Issues
1. **Optuna Import Errors**
   - Ensure optuna is installed: `pip install optuna`
   - Check Python version compatibility

2. **Constraint Violations**
   - Verify parameter naming consistency
   - Check for overlapping fixed/variable parameters
   - Validate constraint key generation

3. **Memory Issues**
   - Reduce batch sizes for large models
   - Use gradient checkpointing for memory efficiency
   - Monitor GPU memory usage

4. **Performance Problems**
   - Adjust batch timeout settings
   - Optimize constraint specification
   - Use appropriate batch sizes

### Debug Commands
```bash
# Enable debug logging
export MODELBATCH_DEBUG=1

# Profile memory usage
python -m memory_profiler examples/optuna_integration_demo.py

# Debug constraint grouping
python -c "
from modelbatch.optuna_integration import ConstraintSpec
spec = ConstraintSpec(...)
print(spec.get_constraint_key(params))
"
```