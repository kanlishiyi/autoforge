# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Core configuration management system
- Bayesian optimization using Optuna
- Grid and random search optimizers
- Experiment tracking with SQLite backend
- Web API with FastAPI
- CLI interface
- Visualization utilities

## [0.1.0] - 2024-01-15

### Added
- **Core Features**
  - Configuration management with YAML/JSON support
  - Configuration inheritance and environment variable substitution
  - Parameter validation using Pydantic

- **Optimization**
  - Bayesian optimization using TPE sampler
  - Grid search for small parameter spaces
  - Random search baseline
  - Early stopping and trial pruning

- **Experiment Tracking**
  - SQLite backend for experiment storage
  - Metric logging and history
  - Artifact management
  - Experiment comparison

- **API & CLI**
  - RESTful API with FastAPI
  - Command-line interface with Click
  - Progress display with Rich

- **Visualization**
  - Learning curve plots
  - Optimization history plots
  - Parameter importance visualization
  - Slice and contour plots

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- Input validation on all API endpoints
- SQL injection prevention in SQLite backend

## [0.2.0] - Planned

### Added
- AI Agent optimization strategy
- Distributed training support
- Model registry and deployment
- W&B and MLflow integration
- Web dashboard

### Changed
- Improved error handling
- Better logging and debugging

## [0.3.0] - Planned

### Added
- Multi-objective optimization
- Async optimization with async objective functions
- Custom sampler support
- Checkpoint resumption
- Cloud storage backends (S3, GCS, Azure)
