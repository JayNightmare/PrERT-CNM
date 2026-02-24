# Changelog

All notable changes to the PrERT-CNM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- Created foundational Month 1 sprint directories: `config/`, `models/`, `engine/`, `tests/`
- Generated baseline `requirements.txt` incorporating AI tracking components (`transformers`, `torch`, `pgmpy`, `datasets`).
- Deployed architectural stubs `privacy_indicators.json`, `privacy_bert.py`, `bayesian_scorer.py`, and `test_pipeline.py` reflecting the sprint roadmap with integrated comments challenging conventional NLP classification architectures.
- Initiated `MEMORY.md` to persist project-specific operational constraints and architectural philosophy.
- Implemented structured JSON data loading with Pydantic for indicator config (`config/loader.py`).
- Extended `PrivacyFeatureExtractor` to include Hugging Face `Trainer` loops (`models/privacy_bert.py`).
- Initialized Bayesian Network graph topologies with discrete CPD structures (`engine/bayesian_scorer.py`).
- Deployed end-to-end geometry and scaling bounds unit tests under `pytest`.
