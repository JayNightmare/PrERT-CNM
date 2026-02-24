"""
Purpose:
Load and validate the `privacy_indicators.json` file against strict Pydantic schemas.

Objective (Month 1, Week 1):
Ensure the probabilistic risk scoring engine (Bayesian Network) operates on guaranteed, structured inputs.

Forward-Thinking / Scalability:
Standard JSON loading is brittle and error-prone. By enforcing structure via Pydantic, 
we establish a strict contract between the configuration files and the risk engine. 
This prevents silent failures if new indicators or frameworks are added dynamically. 
The system will loudly reject improperly formatted risk inputs before any probabilistic computation begins.
"""

from typing import Dict, List, Optional
import json
from pathlib import Path
from pydantic import BaseModel, Field

class AttributeConfig(BaseModel):
    frameworks: List[str] = Field(min_length=1, description="List of framework clauses this attribute satisfies (e.g. 'GDPR:Article_5').")

class CategoryConfig(BaseModel):
    attributes: Dict[str, AttributeConfig] = Field(description="Dictionary of bottom-level attributes mapped to their framework requirements.")

class RootConfig(BaseModel):
    _comment: Optional[str] = None
    categories: Dict[str, CategoryConfig] = Field(description="Top-level privacy categories containing bottom-level attributes.")

def load_config(file_path: str | Path) -> RootConfig:
    """Loads and validates the hierarchical privacy indicators JSON configuration."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return RootConfig(**data)
