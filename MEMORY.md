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

1. **Taxonomy Structuring (Top vs. Bottom Level):** We must define universally applicable _Top-Level Categories_ (e.g., Access Control, Data Retention). Within these, we define the _Bottom-Level Attributes_ (e.g., Password Length, Encryption Standards).
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
- [Completed] Scripted offline caching loader for the OPP-115 alternative mirror dataset (`data/download.py`).
- [Completed] Refactored `privacy_indicators.json`, `loader.py`, and `bayesian_scorer.py` to enforce the Hierarchical Multi-Label Classification mapping (Categories -> Attributes -> Frameworks), pivoting away from the previously flat architecture.
- [Completed] Explicitly extracted and mapped ISO/IEC 27001:2022 Annex A controls and IEEE 7002 clauses into the `privacy_indicators.json` and tightened corresponding topology tests.
- [Completed] Refactored the interactive showcase API (`api/showcase_server.py`) to bypass mock data overrides and dynamically query the PyTorch `PrivacyFeatureExtractor` for authentic risk predictions.
- **[Completed]** Scaffolded Agile Sprint Plan for Month 1 mapping international privacy standards to measurable indicators (`docs/Agile_Sprint_Plan_Month_1.md`).
- **[Completed]** Scaffolded foundational `prert` core architecture module (`prert/`) with stateful Ingestion, DeBERTa Encoder, Attention Explainer, and a unified predictive Pipeline.
- **[Completed]** Upgraded `prert/ingestion.py` to leverage `chromadb` for scalable Context Memory and `PyMuPDF` for authentic semantic PDF extraction, fulfilling Week 1 & 2 ingestion capabilities.
- **[Completed]** Configured DeBERTa-v3 as the core sequence encoder and built out Attention Rollout logic for interpretability, fulfilling Week 3 mechanisms.
- **[Completed]** Implemented a Custom Multi-Label Classification Head in the `PrERTPipeline` to generate explicit predictions over specific ISO standards with confidence scoring.
- **[Completed]** Defined rigorous hierarchical ground truth data schemas (`data/schemas/iso_targets.json`) and implemented a dynamically parsed `Pydantic` `TargetValidator` (`config/target_validator.py`) to enforce target labels and specific class tensor weighting at runtime.
- **[Completed]** Created `tests/generate_synthetic_policies.py` to auto-generate baseline test cases (Perfect, Mixed, Absolute Violation) for explicit unit testing boundary conditions without dataset noise.
- **[Completed]** Refactored `PrERTPipeline` and `AttentionExplainer` to conform to a mathematically rigorous `{"token": str, "weight": float}` JSON structure, perfectly mirroring the model's true attention matrices directly to the frontend.
- **[Completed]** Overhauled Interpretability Engine to utilize Gradient-Weighted Attention (via `logit.backward(retain_graph=True)`) over naive attention rollout. This maps tokens to specific predicted ISO classes and strictly penalizes DeBERTa structural 'sink' routing words.
- **[Completed]** Executed a total paradigm shift from Multi-Label Sequence Classification to Natural Language Inference (NLI) to mitigate generic "Bag of Words" false positives. Extracted contextual hypotheses from `iso_targets.json` schemas, pairing them against text chunks directly in the `PrERTEncoder`. Attention explainer now directly hooks onto the exact contextual gradient of the "Entailment" logit.
- **[Completed]** Substituted final logical classifiers with a full 'Contextual Neural Memory' generative entity (`cnm_agent.py`) utilizing LangChain and HuggingFace Pipelines. Enforced strict Pydantic JSON outputs (`ComplianceReasoning`) demanding natural language Chain-of-Thought mappings before assigning regulatory flags to definitively align with the NIST AI Risk Management Framework goal of a 'Glass-box' architecture.
- **[Completed]** Engineered the Hybrid PrERT-CNM architecture. Reinstalled DeBERTa-v3 as the mathematical Perceptual Layer (to handle entailment likelihood and exact attention token gradients). Configured the Mistral-7B LLM as the Cognitive Layer, invoked strictly on violations to provide concise context combining the raw semantic string via LangChain alongside the DeBERTa `salience_weight` distributions.
- **[Completed]** Hybrid Tri-Color UI and Broad-Context Memory Loop: Successfully overhauled the `Interactive_Showcase.html` interface to display mathematical attention gradients via explicitly colored Red/Green/Blue token CSS matrices. Integrated the semantic `ContextMemoryBank` to serve bounding segment data (chunk-1, chunk, chunk+1) to the API payload, empowering Mistral-7B to perform highly contextual normative analysis on DeBERTa's isolated tokens.
- **[Completed]** Resolved HuggingFace Pipeline warnings related to `max_length` and `max_new_tokens` conflicts during Mistral generation by explicitly overriding the specific model `generation_config`.
- **[Completed]** Refactored inference pipeline into strict 5-step execution sequence integrating Ingest, Encode, Attention Heatmaps, and CNM Reasoning Outputs, optimizing schema for UI consumption.
- **[Completed]** Created advanced fine-tuning scaffold for DeBERTa utilizing K-Fold cross-validation on the OPP-115 dataset to rigorously prevent data leakage during Month 2 training sweeps.
- **[Completed]** Conducted comprehensive Sprint 1 Review and generated `docs/Sprint_1_Review.md`, identifying critical missing regulatory source texts for ingestion.
- **[Completed]** Transitioned PrERT-CNM evaluation logic from DeBERTa Zero-Shot probabilities to Generative Agent-Driven (Mistral LLM) compliance checking to proactively capture complex, absolute compliance violations.
- **[Completed]** Rolled back to Hybrid Architecture. Reinstated DeBERTa-v3 as the strict mathematical gatekeeper, evaluating NLI Entailment logits directly against dynamic ISO hypotheses. Restored Generative Agent (Mistral LLM) strictly to a reasoning explainer triggered _only_ by mathematical violations, restoring explicit confidence bounds and fixing the 100% false-positive collapse caused by the full LLM classification approach.
- **[Completed]** Implemented Agentic Reflection loop. Upgraded CNMAgent into a dual-agent system (Review Agent + Test Agent) to deeply evaluate DeBERTa mathematical flags for false positives and self-correct logic up to 3 attempts, allowing the pipeline to autonomously distinguish between actual violations and contextual noise.

## Next Steps

- Procure authoritative source texts for NIST Privacy Framework, GDPR, and IEEE 7002, and automate batch ingestion via the `prert` pipeline to resolve incomplete Week 1 Task 1.
- Integrate fine-tuning loop results into the broader CI/CD pipeline and evaluate final CV metrics against the >92.5% accuracy target.

## Architectural Decisions

- Decoupled Neural Extraction from Probabilistic Inference: Standard NLP text classifiers suffer from opacity and probability collapse. Using a Bayesian Network on top of the transformer representations forces the engine to explicitly manage uncertainty and causal structures, making it resilient to adversarial legal phrasing. We must remain skeptical of end-to-end differentiable solutions for compliance—explicit probabilistic maps afford auditability required by GDPR and NIST.
- Transitioned to Agent-Driven Compliance Evaluation: The Zero-Shot NLI sequence classification approach proved insufficient for absolute policy violations that relied on nuanced cross-clause context. By forwarding DeBERTa's mathematical attention matrices (heatmaps) and the broader document context to the Mistral generative cognitive agent, the agent now explicitly evaluates compliance (`{"is_compliant": bool}`) through Chain-of-Thought reasoning rather than relying on brittle zero-shot probability thresholds (`prob > 0.5`).
- **Emergency Architecture Rollback (Hybrid Model Restored)**: Pure LLM-driven classification resulted in systemic probability collapse (0% confidence, 100% false-positive violation triggering). We have revoked classification capabilities from the Mistral CNM Agent. DeBERTa-v3 NLI mathematically evaluates True/False states via exact Softmax distribution (`prob > 0.5`). The generative LLM is restricted to a "cognitive explainer" role, invoked _only_ when DeBERTa flags a violation, restoring the strict mathematical foundation necessary for risk quantification.
