# Agile Sprint Plan: Month 1 - Privacy Standard to Indicator Mapping & PrERT Scaffold

## Objective

Map measurable privacy principles from ISO/IEC, NIST, GDPR, IEEE, and international data protection regulations into privacy indicators, while scaffolding the core `prert` pipeline.

## Week 1: Foundational Taxonomy & Regulatory Ingestion

[ ] - **Task 1:** Collate and parse source texts for ISO/IEC 27001/27701, NIST Privacy Framework, GDPR, and IEEE 7002.
[ ] - **Task 2:** Define Top-Level Categories (e.g., Data Minimization, Cryptographic Controls) and Bottom-Level Attributes.
[x] - **Task 3:** Establish the base ingestion pipeline for raw legal texts (PDF/Text) and chunking strategies.

## Week 2: Measurable Indicator Engineering

[ ] - **Task 1:** Map specific regulatory clauses to quantifiable, hierarchical multi-label indicators.
[x] - **Task 2:** Scaffold the Context Memory Bank to maintain state across text chunks during inference.
[x] - **Task 3:** Define ground truth data schemas for model targets (class weighting and augmentation prep).

## Week 3: Model Architecture & Attention Mechanisms

[x] - **Task 1:** Integrate DeBERTa transformer as the core sequence encoder.
[x] - **Task 2:** Design custom task-specific attention heads to apply class weights and extract token-level salience.
[x] - **Task 3:** Implement attention extraction functions for real-time highlighting and heatmap generation.

## Week 4: Pipeline Integration & Audit Trail Generation

[ ] - **Task 1:** Connect the ingestion, DeBERTa encoder, and custom attention mechanisms into a unified inference pipeline.
[ ] - **Task 2:** Format model outputs to yield hierarchical compliance matrices (number of flags, triggered text, ISO controls).
[ ] - **Task 3:** Establish the Audit trail logging mechanism for explainability and non-repudiation of model decisions.
