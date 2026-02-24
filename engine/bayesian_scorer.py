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

from config.loader import RootConfig

class BayesianRiskEngine:
    def __init__(self):
        self.network = DiscreteBayesianNetwork()
        self.inference_engine = None

    def build_topology(self, dependencies: list[tuple[str, str]]):
        """Constructs the DAG from mapped legal standards."""
        self.network.add_edges_from(dependencies)

    def build_topology_from_config(self, config: RootConfig):
        """
        Dynamically generates DAG edges based on the parsed Pydantic settings.
        Maps each attribute (child) to its overarching category (parent)
        and each category to the framework risk composite node.
        """
        edges = []
        frameworks_found = set()
        
        for category_name, category_data in config.categories.items():
            for attribute_name, attribute_config in category_data.attributes.items():
                # Attribute -> Top-level Category
                edges.append((attribute_name, category_name))
                
                # Top-level Category -> Specific Framework Risk (e.g., GDPR_Risk)
                for framework_clause in attribute_config.frameworks:
                    # e.g., 'GDPR:Article_5' -> 'GDPR'
                    framework_name = framework_clause.split(":")[0]
                    frameworks_found.add(framework_name)
                    # We link the attribute to the category, and the category to the framework
                    # In a fully connected DAG: attribute -> category -> framework risk
                    # Prevent duplicate edges
                    edge = (category_name, f"{framework_name}_Risk")
                    if edge not in edges:
                        edges.append(edge)

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
