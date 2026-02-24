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

class IndicatorConfig(BaseModel):
    indicators: List[str] = Field(min_length=1, description="List of measurable privacy indicators for a specific principle.")

class FrameworkConfig(BaseModel):
    # Mapping of Principles (e.g. 'Article_5_Data_Minimization') to their IndicatorConfig
    model_config = {"extra": "allow"}

    def get_principles(self) -> Dict[str, IndicatorConfig]:
        """Returns all dynamically loaded principles."""
        return {
            k: IndicatorConfig(**v) if isinstance(v, dict) else v 
            for k, v in self.model_extra.items() 
            if k != "indicators"
        }

class RootConfig(BaseModel):
    _comment: Optional[str] = None
    frameworks: Dict[str, FrameworkConfig]

def load_config(file_path: str | Path) -> RootConfig:
    """Loads and validates the privacy indicators JSON configuration."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return RootConfig(**data)
