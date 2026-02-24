# PrERT-CNM: AI-Driven Privacy Risk Quantification Engine

"Strengthening User Privacy through AI-Driven Risk Quantification and International Standards Alignment"

## Project Overview

AI-based privacy quantification is an emerging but underdeveloped field. Existing works automate privacy-policy analysis but remain disconnected from international standards and lack quantifiable indicators of user or system risk. This project bridges that gap by developing AI methods to quantify user privacy risks in line with ISO/IEC, NIST, GDPR, and related regulations.

The engine combines a transformer-based feature extractor (PrivacyBERT) and a Bayesian Risk Engine to deliver an auditable, causally-driven privacy risk score, advancing the state-of-the-art in AI-enabled privacy assurance.

## Project Schedule and Deliverables

The project follows a rigorous four-month execution roadmap. Below is the detailed plan for approaching each phase, outlining activities and definitive deliverables.

### Month 1: Standards Mapping

**Plan Strategy:** We will systematically analyze international standards (ISO/IEC, NIST, GDPR, IEEE) and decompose abstract legal principles (e.g., consent, data minimization) into concrete, quantifiable privacy indicators. This phase sets up the deterministic rules engine that our probabilistic models will target.
**Deliverable:** International standards to privacy metrics mapping (manifested as strict validation schemas in `config/privacy_indicators.json` and Pydantic models).

_👉 [View the Month 1 Implementation Report (Code, Config, & Tests)](docs/Sprint_1_Report.md)_

### Month 2: Metrics Definition & Synthetic Data Generation

**Plan Strategy:** We will design and test privacy-risk metrics at the user, system, and organizational levels. Crucially, we will generate synthetic datasets representing adversarial and edge-case compliance failures to ensure our metrics hold up against boundary conditions. A digital ecosystem scope will be conceptualized for future scalability.
**Deliverable:** Draft privacy metrics along with synthetic data ready for model ingestion and testing.

### Month 3: AI Prototype Development

**Plan Strategy:** Integrating the neural perception layer with the probabilistic reasoning layer. We will fine-tune PrivacyBERT on the OPP-115 and Polisis datasets for robust clause classification. The outputs from this transformer network will feed into our Bayesian Network, which will exact risk inferences and emit final composite risk scores.
**Deliverable:** Prototype user privacy quantification AI tool (fully wired `models/` and `engine/` modules).

### Month 4: Validation and Reporting

**Plan Strategy:** The integrated prototype will be benched against both real-world baseline data and the synthetic datasets generated in Month 2. We will validate the mathematical soundness of the uncertainty bounds, system latency, and mapping accuracy, concluding with a comprehensive architectural review.
**Deliverable:** Validated tool and final report.

## License

Please refer to the [LICENSE](LICENSE) file for more information. To understand how to deploy and utilize the system deliverables month over month, consult the [docs/how_to_use.md](docs/how_to_use.md) guide.
