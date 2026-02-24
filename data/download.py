"""
Purpose:
Fetch, process, and persist the OPP-115 privacy policy dataset locally.

Objective (Month 1, Week 2/4):
Establish a reliable truth corpus. Network bandwidth is expensive and transient. 
We must cache data locally (e.g., in Parquet or JSON formats) under `data/raw/` 
so the Bayesian Engine and PrivacyBERT models can train deterministically offline.

Forward-Thinking / Scalability:
Do not rely on implicit HTTP requests deep inside training loops. 
Data ingestion and data modeling are separate concerns. This script ensures that 
any engineer (or continuous integration runner) can deterministically hydrate the 
`data/raw/` folder independently of `models/` execution logic.
"""

import os
from pathlib import Path
from datasets import load_dataset

def fetch_and_cache_opp115(output_dir: str = "data/raw"):
    """Downloads the OPP-115 dataset and saves it in a persistent binary format."""
    print("Initiating dataset retrieval for OPP-115...")
    dataset = load_dataset("alzoubi36/opp_115")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Persisting datasets to {output_path.absolute()}")
    dataset.save_to_disk(str(output_path))
    print("Download and local serialization complete.")

if __name__ == "__main__":
    fetch_and_cache_opp115()
