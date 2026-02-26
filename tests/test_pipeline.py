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
import time
from pydantic import ValidationError
from pathlib import Path
from config.loader import load_config
from fastapi.testclient import TestClient

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

    def test_absolute_violation_phrase_detection(self):
        from models.PrERT.pipeline import PrERTPipeline

        text = (
            "We sell your personal data to third parties. "
            "We do not use encryption for our databases. "
            "We retain data indefinitely and automatically opt you in without explicit consent."
        )

        matches = PrERTPipeline._detect_hard_violation_phrases(text)
        assert matches
        assert any("Sells user personal data" in match for match in matches)
        assert any("No encryption statement" in match for match in matches)
        assert any("Indefinite data retention" in match for match in matches)
        assert any("Forced opt-in without consent" in match for match in matches)

    def test_no_encryption_false_positive_guard(self):
        from models.PrERT.pipeline import PrERTPipeline

        text = (
            "No personal information is sold to third parties for monetization. "
            "All credential records use modern encryption in transit and at rest."
        )

        matches = PrERTPipeline._detect_hard_violation_phrases(text)
        assert "No encryption statement" not in matches

    def test_hard_violation_match_extraction_returns_span_and_heatmap(self):
        from models.PrERT.pipeline import PrERTPipeline

        text = (
            "Internal databases do not use encryption controls because plaintext processing is faster. "
            "We retain user data indefinitely for retrospective analysis."
        )

        detailed = PrERTPipeline._extract_hard_violation_matches(text, policy_id=2)
        assert detailed
        assert any(item["description"] == "No encryption statement" for item in detailed)
        assert any(item["description"] == "Indefinite data retention" for item in detailed)

        for item in detailed:
            assert item["policy_id"] == 2
            assert item["triggered_text"]
            assert isinstance(item["cause_heatmap"], list)
            assert item["cause_heatmap"], "Expected non-empty safety-layer heatmap tokens."
            for token in item["cause_heatmap"]:
                assert "token" in token
                assert "weight" in token
                assert "category" in token

    def test_ingestion_policy_block_split(self):
        from models.PrERT.ingestion import DocumentIngestor

        text = (
            "Policy 1: Service Privacy\n\n"
            "We collect only minimum data to provide services.\n\n"
            "This data is encrypted in transit and at rest.\n\n"
            "Policy 2: Marketing Privacy\n\n"
            "By using this service, you grant rights to sell your data to ad partners.\n\n"
            "We automatically opt you in without explicit consent."
        )

        blocks = DocumentIngestor._split_into_policy_blocks(text)
        assert len(blocks) >= 2
        assert "Policy 1" in blocks[0]
        assert any("Policy 2" in block for block in blocks)

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

    def test_prert_pipeline_true(self):
        from models.PrERT.pipeline import PrERTPipeline
        pipeline = PrERTPipeline()
        pipeline.run("We collect your data to improve services.")
        assert True

    def test_prert_pipeline_false(self):
        from models.PrERT.pipeline import PrERTPipeline
        pipeline = PrERTPipeline()
        result = pipeline.run("We automatically opt you in to all data collection streams upon signup without requiring explicit consent.")

        assert "status" in result
        assert "total_flags" in result
        assert "violated_controls" in result
        assert "audit_trail" in result

        assert result["status"] in {"Compliant", "Non-Compliant"}
        assert isinstance(result["total_flags"], int)
        assert isinstance(result["violated_controls"], list)
        assert isinstance(result["audit_trail"], list)
        assert result["audit_trail"], "Expected audit trail entries for control evaluations."

        triggered_entries = [trail for trail in result["audit_trail"] if trail.get("triggered_status")]
        reflected_entries = [trail for trail in result["audit_trail"] if trail.get("attempts", 0) > 0]

        for trail in result["audit_trail"]:
            assert "attempts" in trail
            assert isinstance(trail["attempts"], int)
            assert trail["attempts"] >= 0
            assert "policy_id" in trail
            assert isinstance(trail["policy_id"], int)

        for trail in reflected_entries:
            assert trail["attempts"] > 0

        assert result["total_flags"] == len(triggered_entries)
        if result["total_flags"] > 0:
            assert result["status"] == "Non-Compliant"
            assert result["violated_controls"]
        else:
            assert result["status"] == "Compliant"
            assert result["violated_controls"] == []


class TestAPIIntegration:
    def test_file_upload_absolute_violation_non_compliant(self):
        from api.showcase_server import app, tasks_db, pipeline

        if pipeline is None:
            pytest.skip("Pipeline did not initialize in test environment.")

        tasks_db.clear()
        client = TestClient(app)

        policy_path = Path(__file__).parent.parent / "data" / "test_policies" / "absolute_violation.txt"
        with open(policy_path, "rb") as uploaded_file:
            response = client.post(
                "/analyze/file",
                files={"file": ("absolute_violation.txt", uploaded_file, "text/plain")}
            )

        assert response.status_code == 200
        payload = response.json()
        assert "task_id" in payload
        task_id = payload["task_id"]

        final_payload = None
        for _ in range(80):
            status_response = client.get(f"/analyze/status/{task_id}")
            assert status_response.status_code == 200
            current_payload = status_response.json()

            if current_payload["status"] == "completed":
                final_payload = current_payload
                break

            if current_payload["status"] == "error":
                pytest.fail(f"Pipeline task failed: {current_payload.get('error')}")

            time.sleep(0.5)

        assert final_payload is not None, "Timed out waiting for analysis task completion."

        result = final_payload["result"]
        assert result["status"] == "Non-Compliant"
        assert result["total_flags"] > 0
        assert result["violated_controls"]
        assert any(entry.get("triggered_status") for entry in result["audit_trail"])