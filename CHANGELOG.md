# Changelog

All notable changes to the PrERT-CNM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Fixed

- Suppressed transformers `max_length` vs `max_new_tokens` warning during text generation by explicitly unsetting `max_length` in the HuggingFace Pipeline's model generation config in `prert/pipeline.py`.

### Changed

- **[Architecture Rollback & Selective Inference Gating]** Restored the Hybrid Architecture where DeBERTa-v3 acts as the primary logical classifier evaluating strict ISO hypotheses via NLI entailment matching. Rewrote `pipeline.py` to enforce strict operational sequences where Heatmap Extraction and Agent Reasoning are only initiated for clauses that DeBERTa formally calculates as violations (`prob > 0.5`).
- **[CNM Agent Inference Capping]** Rescinded Mistral's classification rights in `cnm_agent.py` by removing the `is_compliant` boolean array. The LangChain prompt explicitly forces purely reasoning outputs (_why did the mathematical model trigger a violation?_) to strictly mitigate false-positive regulatory noise.
- **[Rigorous Pipeline Execution]** Rewrote `prert/pipeline.py` to strip inline comments and strictly guarantee a traceable 5-step sequential flow: Ingest -> Encode (DeBERTa) -> Extract Attention (Heatmap) -> CNM Agent Reasoning (Mistral) -> Finish/Output. Explicitly logs each state transition for observability.
- Refactored `prert/pipeline.py` to calculate explicit Pass/Fail states for all mapped ISO controls regardless of violation.
- Expanded the `TokenHighlight` engine to bin normalized DeBERTa attention gradients into categorical Red (High Risk), Green (Contextual), and Blue (Safe) token buckets.
- Injected wider `broader_context` document slices directly from the `ContextMemoryBank` into the Mistral `PromptTemplate` to enhance LLM reasoning constraints.
- Overhauled `showcase_server.py` API schema (`AuditTrailEntry`) to surface boolean compliance states.
- Rewrote `Interactive_Showcase.html` to visualize data using HTML5 `<details>` standard accordions, grouping green ticks and red crosses dynamically alongside the CSS token grid.

### Added

- **[Actionable Schema]** Overhauled `prert/pipeline.py` return dicts. The `Compliance Evaluation Results` now explicitly surfaces `status`, `total_flags`, and `violated_controls` (with mathematical classification confidence). Features restructured `audit_trail` outputs explicitly demanding UI-ready `triggered_status`, `cnm_reason`, and `cause_heatmap` token definitions mapping exactly to Tri-Color matrix UI logic.
- **[Leak-Proof Finetuning Engine]** Scaffolded `prert/finetune.py` defining an advanced K-Fold cross-validation DeBERTa training loop over the `coastalcph/opp-115` repository explicitly structured to prevent overlapping context data leakage across train/validation folds.

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
- **[Interpretability System Overhaul]**: Bypassed generic sequence Attention Rollout in favor of class-specific Gradient-Weighted Attention (`logit.backward()`). Features native stop-word routing penalization and mathematically isolates exactly which clause fragments triggered individual individual normative breaches (ISO/GDPR mappings).
- **[Architecture Paradigm Shift]**: Completely deprecated standard multi-label sequence classification `MultiLabelClassificationHead` inside `PrERTPipeline` due to high false-positive 'Bag of Words' vulnerability. Forcibly restructured the core `PrERTEncoder` and downstream pipeline into a Natural Language Inference (NLI) machine, chaining explicit failure hypotheses to original document streams to interrogate DeBERTa's explicit contextual entailment graphs.
- **[Generative Agent Layer]**: Initialized `cnm_agent.py` acting natively inside the LangChain ecosystem to run unstructured generation models. Bypassed binary probability masking arrays forcing explicit textual hypothesis explanation mapping into the backend REST API payload schema underneath a `thought_process` node.
- **[Hybrid Two-Staged Architecture]**: Re-enabled the `NLIClassificationHead` and `AttentionExplainer` loops in the pipeline to restore accurate statistical token maps. Updated `CNMAgent`'s LangChain templates to operate exclusively as a cognitive describer bridging DeBERTa's gradient outputs rather than serving as the baseline text classifier.
