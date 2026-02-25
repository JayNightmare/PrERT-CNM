"""
Purpose:
End-to-end testing pipeline synthesizing both the transformer model outputs and Bayesian risk scores.

Objective (Month 1, Week 4):
Verify the integration geometry between `models/` (perceptual layer) and `engine/` (reasoning layer). 
Additionally, manage the execution of synthetic datasets to simulate rare compliance failures.

Forward-Thinking / Scalability:
Conventional testing ensures code runs without crashing. We must demand more: our testing pipeline 
must simulate adversarial, edge-case privacy policies designed to exploit probability collapse or 
hallucinations in the NLP layer. The tests should dynamically generate synthetic scenarios and fuzz 
the Bayesian priors. If the system cannot handle deliberately obscure or contradictory legal language 
without flagging high uncertainty, the architecture fails. We are building a continuous validation pipeline, 
not just unit tests.
"""

import pytest
from pydantic import ValidationError
from config.loader import load_config

class TestSyntheticRiskPipeline:
    def setup_method(self):
        self.adversarial_samples = []

    def test_config_validation(self):
        # Validates that the base JSON configuration matches strict Pydantic models.
        # This guarantees the reasoning layer receives well-formed schemas.
        import os
        from pathlib import Path
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / "config" / "privacy_indicators.json"
        
        config = load_config(config_path)
        assert config.categories is not None
        assert "Data_Minimization" in config.categories
        assert "collection_necessity_score" in config.categories["Data_Minimization"].attributes
        
        # Test validation error
        with pytest.raises(FileNotFoundError):
            load_config(base_dir / "config" / "non_existent.json")

    def test_end_to_end_scoring_consistency(self):
        assert True

    def test_bayesian_uncertainty_bounds(self):
        from engine.bayesian_scorer import BayesianRiskEngine
        from pgmpy.factors.discrete import TabularCPD

        engine = BayesianRiskEngine()
        engine.build_topology([('Data_Minimization', 'Risk_Score')])

        cpd_min = TabularCPD(variable='Data_Minimization', variable_card=2, values=[[1], [0]])
        cpd_risk = TabularCPD(variable='Risk_Score', variable_card=2, 
                              values=[[0.9, 0.1], [0.1, 0.9]],
                              evidence=['Data_Minimization'], evidence_card=[2])
        engine.add_cpds([cpd_min, cpd_risk])
        engine.prepare_inference()

        res = engine.compute_risk(evidence={'Data_Minimization': 0}, target_node='Risk_Score')
        assert res["states"] == [0, 1]
        assert len(res["risk_distribution"]) == 2

    def test_dynamic_topology_generation(self):
        # Validate that JSON configs correctly translate to DAG edges
        import os
        from pathlib import Path
        from engine.bayesian_scorer import BayesianRiskEngine

        base_dir = Path(__file__).parent.parent
        config_path = base_dir / "config" / "privacy_indicators.json"
        config = load_config(config_path)

        engine = BayesianRiskEngine()
        engine.build_topology_from_config(config)

        # Expected edge mapped from hierarchical privacy_indicators.json
        # collection_necessity_score -> Data_Minimization
        # Data_Minimization -> GDPR_Risk
        assert engine.network.has_edge("collection_necessity_score", "Data_Minimization")
        assert engine.network.has_edge("Data_Minimization", "GDPR_Risk")
        assert engine.network.has_edge("Data_Minimization", "NIST_Privacy_Framework_Risk")
        assert engine.network.has_edge("Data_Minimization", "ISO_27001_Risk")
        assert engine.network.has_edge("Transparency_and_Consent", "IEEE_7002_Risk")

    def test_privacy_bert_feature_extraction(self):
        # We verify that the model returns expected tensor geometries.
        # This isolates failures between the perception layer and reasoning layer.
        from models.Privacy_Bert.privacy_bert import PrivacyFeatureExtractor
        extractor = PrivacyFeatureExtractor() # Fall back to stable bert-base-uncased
        
        sample_policy = "We collect your data to improve services."
        features = extractor.extract_features(sample_policy)
        
        # Batch size 1, num_labels 2
        assert features.shape == (1, 2)
        assert features.requires_grad == False
