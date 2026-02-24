# Project Memory: PrERT-CNM

## Identity & Role
AI-Driven Privacy Risk Quantification Engine. Unifies transformer-based policy extraction (PrivacyBERT) with probabilistic risk modeling (Bayesian Networks). Aligns unstructured legal text against international standards (ISO/IEC, NIST, GDPR) using quantifiable indicators.

## Current Project State
The project architecture and Month 1 sprint map are actively being scaffolded. 

- **Sprin Roadmap (Month 1):**
  - Week 1: Map measurable privacy principles into quantifiable indicators (`config/`).
  - Week 2: Fine-tune PrivacyBERT on OPP-115 and Polisis datasets (`models/`).
  - Week 3: Build the Bayesian risk scoring engine using pgmpy (`engine/`).
  - Week 4: Generate synthetic datasets and build the testing pipeline (`tests/`).

## Activity Requirements: Month 1 Deep Analysis
**Objective**: Map measurable privacy principles from ISO/IEC, NIST, GDPR, IEEE and international data protection regulations into privacy indicators.

**Analysis of Current State vs. Actual Requirements**:
Our initial execution scaffolded a basic JSON representation mapping abstract GDPR concepts (e.g., Article 5) to placeholder indicators. However, reviewing `docs/Architecture-Stack.md` and `docs/Model.md`, this is entirely insufficient. The final PrERT-CNM (Privacy BERT Contextual Neural Memory) model requires a **Hierarchical Multi-Label Classification** system (Top Level: High-level ISO domain; Bottom Level: Fine-grained requirements like Encryption Standards).

Therefore, Month 1's true deliverable must structurally align with this 2-stage model design. 

**Deep Work Breakdown for Month 1:**
1. **Taxonomy Structuring (Top vs. Bottom Level):** We must define universally applicable *Top-Level Categories* (e.g., Access Control, Data Retention). Within these, we define the *Bottom-Level Attributes* (e.g., Password Length, Encryption Standards).
2. **Cross-Framework Overlays (Universal Schema):** The mapping must unify ISO/IEC, NIST, GDPR, and IEEE under communal privacy indicators. A single fine-grained attribute must link to its specific clause in GDPR (Art 32) and NIST AI RMF simultaneously.
3. **Measurability & Scoring Bounds:** Indicators must be inherently quantifiable (e.g., Boolean existence flags or probability distributions) to properly parameterize the Bayesian Risk engine.
4. **Data Structure Overhaul (`config/privacy_indicators.json`):** The current JSON loader expects a flat list of indicators per specific framework principle. This must be entirely refactored into a hierarchical knowledge graph that the CNM (Contextual Neural Memory) can traverse to apply specialized, fine-grained context rules.

**Pivot Strategy:** The `config` module mapping logic must be completely rewritten. We need to draft a comprehensive schema reflecting the Multi-Label Hierarchical requirements before moving on to Month 2.

## Active Tasks
- Initialized core project directories: `config/`, `models/`, `engine/`, `tests/`
- Created `requirements.txt` with AI infrastructure dependencies (`transformers`, `torch`, `pgmpy`, `datasets`).
- Deployed boilerplate implementations with critical architectural commentary questioning standard approaches.
- **[Completed]** Implemented structured JSON data loading with Pydantic (`config/loader.py`).
- **[Completed]** Extended `PrivacyFeatureExtractor` to include Hugging Face `Trainer` loops (`models/privacy_bert.py`).
- **[Completed]** Initialized Bayesian Network graph topologies with CPD integration logic (`engine/bayesian_scorer.py`).
- **[Completed]** Implemented end-to-end integration boundaries via `pytest` (`tests/test_pipeline.py`).
- **[Completed]** Built dynamic JSON to DAG topology parser to ensure GDPR configs directly dictate DAG relationships (`engine/bayesian_scorer.py`).
- **[Completed]** Scripted offline caching loader for the OPP-115 alternative mirror dataset (`data/download.py`).

## Next Steps
- Execute full training sweep for Month 2 using the fetched OPP-115 corpus and the `PrivacyFeatureExtractor`.

## Architectural Decisions
- Decoupled Neural Extraction from Probabilistic Inference: Standard NLP text classifiers suffer from opacity and probability collapse. Using a Bayesian Network on top of the transformer representations forces the engine to explicitly manage uncertainty and causal structures, making it resilient to adversarial legal phrasing. We must remain skeptical of end-to-end differentiable solutions for compliance—explicit probabilistic maps afford auditability required by GDPR and NIST.
