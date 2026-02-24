from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path
import random

from config.loader import load_config
from engine.bayesian_scorer import BayesianRiskEngine
from pgmpy.factors.discrete import TabularCPD
import torch
from models.privacy_bert import PrivacyFeatureExtractor

app = FastAPI(title="PrERT-CNM Interactive Showcase API", version="1.0")

# Allow the frontend (HTML) to talk to this local server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Global Pipeline State ---
base_dir = Path(__file__).parent.parent
config_path = base_dir / "config" / "privacy_indicators.json"

try:
    config = load_config(config_path)
    engine = BayesianRiskEngine()
    engine.build_topology_from_config(config)
    
    print("Loading PyTorch Transformer Model... (PrivacyBERT)")
    extractor = PrivacyFeatureExtractor()
    print("Transformer loaded successfully.")
    
    # Generate mock Bayesian CPDs for the demonstration graph
    # In a real environment, these probabilities would be learned via maximum likelihood
    # from empirical datasets (e.g., OPP-115).
    # For this showcase, we explicitly set logical priors.
    cpds = []
    
    # Attributes (Priors derived from mock "Neural Extraction" certainty)
    all_attributes = []
    for cat in config.categories.values():
        for attr in cat.attributes.keys():
            all_attributes.append(attr)
            
    frameworks = set()
    for cat_name, cat_data in config.categories.items():
        for attr_name, attr_config in cat_data.attributes.items():
            for f in attr_config.frameworks:
                frameworks.add(f.split(":")[0])
                
    # We'll use a simplified mock engine for the interactive showcase. 
    # Exact tabular CPD creation over large dynamic topologies requires exhaustive state permutations 
    # (e.g., 2 parents = 4 parameter columns). 
    # To keep the showcase lightweight and strictly focused on demonstrating the DAG propagation 
    # geometry (Attributes -> Categories -> Frameworks), we'll mock the risk scoring calculation mathematically 
    # outside of pgmpy for the *showcase endpoint*, while proudly displaying the structural DAG we mapped.

except Exception as e:
    print(f"Error initializing backend: {e}")

class PolicyAnalysisRequest(BaseModel):
    text: str

class AnalysisResult(BaseModel):
    attributes_triggered: dict[str, float]
    category_risks: dict[str, float]
    framework_risks: dict[str, float]
    dag_edges: list[tuple[str, str]]

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_policy(request: PolicyAnalysisRequest):
    """
    Mock endpoint simulating the full PrERT-CNM pipeline.
    1. Simulates Transformer text extraction (Attribute Triggers).
    2. Simulates Bayesian Risk Propagation (Hierarchical Risk).
    """
    text = request.text.lower()
    
    # 1. Perception Layer: REAL Transformer Extractions
    # Running the actual text through our PrivacyBERT class:
    outputs = extractor.extract_features(text)
    
    # Convert logits (raw dimensions) into a probability distribution [Safe, Risky]
    probs = torch.nn.functional.softmax(outputs, dim=1).squeeze().tolist()
    
    # Extract the 'Risk' probability from standard BERT output structure
    transformer_risk = probs[1] if len(probs) > 1 else sum(probs) / len(probs)
    transformer_safe = 1.0 - transformer_risk

    # Mapping the true transformer global risk + contextual lexical scanning 
    # to trigger the fine-grained hierarchical attributes.
    attrs = {
        "collection_necessity_score": max(0.1, transformer_safe) if "collect" in text else 0.8,
        "retention_longevity_factor": max(0.1, transformer_safe) if "retain" in text or "keep" in text else 0.9,
        "opt_in_clarity_index": 0.9 if ("consent" in text or "agree" in text) and transformer_risk < 0.6 else 0.3,
        "privacy_notice_readability": 0.8 if "understand" in text and transformer_risk < 0.6 else 0.4,
        "withdrawal_friction_estimate": 0.2 if "withdraw" in text and transformer_risk > 0.5 else 0.7,
        "encryption_at_rest": 0.9 if "encrypt" in text or "secure" in text else 0.1,
    }

    # 2. Reasoning Layer: Propagate Risk Upwards
    category_risks = {}
    for cat_name, cat_data in config.categories.items():
        # A category's risk is the inverted average of its attribute "health" scores
        # We simulate Bayesian propagation where 1 failing attribute spikes the category risk.
        cat_attr_scores = [attrs[a] for a in cat_data.attributes.keys()]
        if cat_attr_scores:
            # High attribute score = healthy. Low attribute score = risky.
            risk = 1.0 - (sum(cat_attr_scores) / len(cat_attr_scores))
            category_risks[cat_name] = round(risk, 2)

    # 3. Framework Mapping
    framework_risks = {fw: 0.0 for fw in frameworks}
    for cat_name, cat_data in config.categories.items():
        for attr_name, attr_config in cat_data.attributes.items():
             for fw_clause in attr_config.frameworks:
                 fw_name = fw_clause.split(":")[0]
                 # The framework takes the worst-case risk from its linked attributes/categories
                 attr_risk = 1.0 - attrs[attr_name]
                 if attr_risk > framework_risks[fw_name]:
                     framework_risks[fw_name] = round(attr_risk, 2)

    # Clean up attributes to probabilities for display
    attrs = {k: round(v, 2) for k, v in attrs.items()}

    return AnalysisResult(
        attributes_triggered=attrs,
        category_risks=category_risks,
        framework_risks=framework_risks,
        dag_edges=list(engine.network.edges())
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "categories_loaded": len(config.categories.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
