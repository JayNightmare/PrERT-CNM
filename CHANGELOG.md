# Changelog

All notable changes to the PrERT-CNM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Scaffolded Month 1 Agile Sprint Plan mapping international privacy standards to metrics (`docs/Agile_Sprint_Plan_Month_1.md`).
- Scaffolded foundational `prert` pipeline module (`prert/ingestion.py`, `encoder.py`, `attention.py`, `pipeline.py`) incorporating context memory banks and transparent attention extraction logic.
- Created foundational Month 1 sprint directories: `config/`, `models/`, `engine/`, `tests/`
- Generated baseline `requirements.txt` incorporating AI tracking components (`transformers`, `torch`, `pgmpy`, `datasets`).
- Deployed architectural stubs `privacy_indicators.json`, `privacy_bert.py`, `bayesian_scorer.py`, and `test_pipeline.py` reflecting the sprint roadmap with integrated comments challenging conventional NLP classification architectures.
- Initiated `MEMORY.md` to persist project-specific operational constraints and architectural philosophy.
- Implemented structured JSON data loading with Pydantic for indicator config (`config/loader.py`).
- Extended `PrivacyFeatureExtractor` to include Hugging Face `Trainer` loops (`models/privacy_bert.py`).
- Initialized Bayesian Network graph topologies with discrete CPD structures (`engine/bayesian_scorer.py`).
- Deployed end-to-end geometry and scaling bounds unit tests under `pytest`.
- Activated dynamic topology mappings correlating Pydantic JSON logic to explicit `pgmpy` mathematical DAG structures (`engine/bayesian_scorer.py`).
- Cached and persisted the alternative public `OPP-115` policy corpus offline onto the disk (`data/download.py`).

### Changed

- **[Breaking]** Refactored `config/privacy_indicators.json`, `loader.py`, and `bayesian_scorer.py` to implement the required Hierarchical Multi-Label Classification tracking. Attributes now strictly map dynamically outward to multiple framework constraints under specific high-level category wrappers.
- Upgraded `ContextMemoryBank` in `prert/ingestion.py` from an ephemeral Python list to a scalable `chromadb` Vector Database.
- Replaced mocked string extraction with `PyMuPDF` for robust PDF ingestion, and implemented semantic chunking to preserve legal boundaries over standard length-based cutting.
- Upgraded `AttentionExplainer` to natively support Attention Rollout mechanisms, extracting layered token salience without relying on naive averaging.
- Integrated a `MultiLabelClassificationHead` into `PrERTPipeline` to independently learn and apply ISO control flags over the DeBERTa hidden states.
- Established a Pydantic-validated ground truth schema architecture (`config/target_validator.py` and `data/schemas/iso_targets.json`) supporting explicit hierarchy (Category -> Attribute) mapping coupled with regulatory framework tracing and embedded classification weights.
- Auto-generated semantic synthetic policy benchmarking files (`tests/generate_synthetic_policies.py`) to map explicit edge-case failures without reliance on public dataset noise.
- Renamed and structurally aligned `AttentionExplainer` and frontend pipeline payloads to uniformly utilize `weight` over `salience` ensuring deterministic API ingestion behavior.
- **[Interpretability System Overhaul]**: Bypassed generic sequence Attention Rollout in favor of class-specific Gradient-Weighted Attention (`logit.backward()`). Features native stop-word routing penalization and mathematically isolates exactly which clause fragments triggered individual normative breaches (ISO/GDPR mappings).
