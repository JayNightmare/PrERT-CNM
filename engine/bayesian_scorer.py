"""
Purpose:
This module houses the probabilistic risk scoring engine driven by Bayesian Networks.

Objective (Month 1, Week 3):
Construct a directed acyclic graph (DAG) mapping the neural features from PrivacyBERT to 
concrete, quantifiable privacy indicators.

Forward-Thinking / Scalability:
Standard risk matrices are ad-hoc, linear, and fail under uncertainty. By treating privacy
risk dynamically via Bayesian inference, we force the system to articulate its confidence.
A key architectural principle here is to question static topologies. The graph structure itself 
must eventually be learnable (causal discovery), or dynamically assembled based on the active 
legal frameworks (e.g., GDPR vs. CCPA). We rely on `pgmpy` to exact posterior probabilities, 
making our final risk scores fully auditable and mathematically consistent.
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

class BayesianRiskEngine:
    def __init__(self):
        self.network = DiscreteBayesianNetwork()
        self.inference_engine = None

    def build_topology(self, dependencies: list[tuple[str, str]]):
        """Constructs the DAG from mapped legal standards."""
        self.network.add_edges_from(dependencies)

    def build_topology_from_config(self, config):
        """
        Dynamically generates DAG edges based on the parsed Pydantic settings.
        Maps each indicator (child) to its overarching principle (parent).
        """
        edges = []
        for framework_name, framework_data in config.frameworks.items():
            for principle_name, indicator_config in framework_data.get_principles().items():
                for indicator in indicator_config.indicators:
                    edges.append((indicator, principle_name))
                
                # Link all principles back to a single composite risk node for the framework
                edges.append((principle_name, f"{framework_name}_Risk"))

        self.build_topology(edges)
        
    def add_cpds(self, cpds: list[TabularCPD]):
        """Injects Conditional Probability Distributions into nodes."""
        self.network.add_cpds(*cpds)
        assert self.network.check_model(), "Invalid Bayesian Network topology or CPDs."
        
    def prepare_inference(self):
        """Initializes the Variable Elimination engine."""
        self.inference_engine = VariableElimination(self.network)

    def compute_risk(self, evidence: dict, target_node: str) -> dict:
        """Executes exact inference given neural outputs as evidence."""
        if not self.inference_engine:
            raise ValueError("Inference engine not prepared. Call prepare_inference() first.")
            
        result = self.inference_engine.query(variables=[target_node], evidence=evidence)
        return {
            "risk_distribution": result.values.tolist(),
            "states": result.state_names[target_node]
        }
