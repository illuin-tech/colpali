# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased

### Added

### Changed

- Remove dummy image from `ColPaliProcessor.process_queries`

### Fixed

## [0.3.1] - 2024-09-27

### Added

- Add module-level imports for collators
- Add sanity check in the run inference example script
- Add E2E test for ColPali
- Add Qwen2-VL support

### Changed

- Improve code clarity the run inference example script
- Subset the example dataset in the run inference example script
- Rename scorer test to `test_processing_utils`
- Greatly simplify routing logic in Trainer selection and when feeding arguments to the model forward pass (refacto)
- Removed class `ContrastiveNegativeTrainer` which is now just integrated in ContrastiveTrainer. This should not affect the user-facing API.
- Bumped transformers version to 4.45.0 to get Qwen2-VL support

### Fixed

- Import HardNegCollator at module-level if and only if datasets is available
- Remove the need for `typer` in the run inference example script
- Fix edge case when empty suffix `""` given to processor
- Fix bug in HardNegCollator since 0.3.0

## [0.3.0] - 2024-09-10

✨ This release is an exhaustive package refacto, making ColPali more modular and easier to use.

🚨 It is **NOT** backward-compatible with previous versions.

### Added

- Restructure the `utils` module
- Restructure the model training code
- Add custom `Processor` classes to easily process images and/or queries
- Enable module-level imports
- Add scoring to processor
- Add `CustomRetrievalEvaluator`
- Add missing typing
- Add tests for model, processor, scorer, and collator
- Lint `Changelog`
- Add missing docstrings
- Add "Ruff" and "Test" CI pipelines

### Changed

- Restructure all modules to closely follow the [`transformers`](https://github.com/huggingface/transformers) architecture
- Hugely simplify the collator implementation to make it model-agnostic
- `ColPaliProcessor`'s `process_queries` doesn't need a mock image input anymore
- Clean `pyproject.toml`
- Loosen the required dependencies
- Replace `black` with the `ruff` linter

### Removed

- Remove `interpretability` and `eval_manager` modules
- Remove unused utils
- Remove `TextRetrieverCollator`
- Remove `HardNegDocmatixCollator`

### Fixed

- Fix wrong PIL import
- Fix dependency issues

## [0.2.2] - 2024-09-06

### Fixed

- Remove forced "cuda" usage in Retrieval Evaluator

## [0.2.1] - 2024-09-02

Patch query preprocessing helper function disalignement with training scheme.

### Fixed

- Add 10 extra pad token by default to the query to act as reasoning buffers. This was added in the collator but not the external helper function for inference purposes.

## [0.2.0] - 2024-08-29

Large refactoring to adress several issues and add features. This release is not backward compatible with previous versions.
The models trained under this version will exhibit degraded performance if used with the previous version of the code and vice versa.

[Branch](https://github.com/illuin-tech/colpali/pull/23)

### Added

- Added multiple training options for training with hard negatives. This leads to better model performance !
- Added options for restarting training from a checkpoint.

### Changed

- Optionally load ColPali models from pre-initialized backbones of the same shape to remove any stochastic initialization when loading adapters. This fixes [11](https://github.com/illuin-tech/colpali/issues/11) and [17](https://github.com/illuin-tech/colpali/issues/17).

### Fixed

- Set padding side to right in the tokenizer to fix misalignement issue between different query lengths in the same batch. Fixes [12](https://github.com/illuin-tech/colpali/issues/12)
- Add 10 extra pad token by default to the query to act as reasoning buffers. This enables the above fix to be made without degrading performance and cleans up the old technique of using `<unused>` tokens.

## [0.1.1] - 2024-08-28
  
Minor patch release to fix packaging issues.

### Fixed

- [Branch](https://github.com/illuin-tech/colpali/commit/bd55e88c7af7069dde943f00665181fb94631cdd)
  Fix .gitignore to include all necessary files in the package.

## [0.1.0] - 2024-08-28

Initial code release corresponding to the paper.
