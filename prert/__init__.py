"""
Purpose: Initialize the PrERT model package.
Goal: Expose the primary pipeline entry point for external consumers.
Scalability & Innovation: Encapsulates internal complexities, providing a clean boundary between the inference engine and the REST/RPC API layers.
"""
from .pipeline import PrERTPipeline
