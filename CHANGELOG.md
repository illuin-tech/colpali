# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased

### Added

- feat: Deprecate `interpretability` and `eval_manager` modules
- feat: Deprecate unused util modules
- feat: Revamp module organization
- feat: Restructure the `utils` module
- feat: Move `ColModelTraining` module
- feat: Lint code + tweaks
- feat: deprecated a lot of unused modules and legacy code

### Changed

- doc: Lint Changelog
- doc: Tweak README
- feat: The processing function in `colpali_engine.utils.processing_utils.colpali_processing_utils` `process_queries` has a changed API and does not require a Mock Image anymore.

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
