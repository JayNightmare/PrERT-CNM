"""
Purpose: Orchestrate the PrERT-CNM pipeline from ingestion through encoding to final hierarchical ISO control mapping.
Goal: Provide a unified interface that returns multi-label compliance evaluations, complete with audit trails, flag counts, and trigger mappings.
Scalability & Innovation: This is not a simple linear pipeline. It represents a stateful compliance machine. By chaining the Context Memory Bank with DeBERTa embeddings and our custom Attention Explainer, we construct a deterministic, traceable evaluation graph that outputs structured risk data, completely divorcing ourselves from opaque end-to-end classification paradigms.
"""
import json
import torch
import torch.nn as nn
from .ingestion import DocumentIngestor
from .encoder import PrERTEncoder
from .attention import AttentionExplainer

class MultiLabelClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features[:, 0, :]  # CLS token state
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return torch.sigmoid(x)

class PrERTPipeline:
    def __init__(self):
        self.ingestor = DocumentIngestor()
        self.encoder = PrERTEncoder()
        
        # Hardcoded class weights representing normative bias toward specific standards
        self.explainer = AttentionExplainer(class_weights={"iso_27001": 1.5, "gdpr": 1.2})
        
        # Baseline explicit mapping representation to link flags identically to regulatory schemas
        self.iso_mapping = {
            "data_retention": "ISO/IEC 27001:2022 A.8.10",
            "encryption": "ISO/IEC 27001:2022 A.8.24"
        }
        self.num_labels = len(self.iso_mapping)
        self.label_keys = list(self.iso_mapping.keys())
        
        self.classification_head = MultiLabelClassificationHead(
            hidden_size=self.encoder.model.config.hidden_size, 
            num_labels=self.num_labels
        ).to(self.encoder.device)

    def process_document(self, content: bytes | str, is_pdf: bool = False) -> dict:
        if is_pdf:
            memory = self.ingestor.process_pdf(content)
        else:
            memory = self.ingestor.process_text(str(content))
            
        audit_trail = []
        total_flags = 0
        
        for segment_data in memory.retrieve_context():
            text = segment_data["segment"]
            
            # 1. Encode into context-aware representations
            hidden_states, attentions, inputs = self.encoder.encode(text)
            
            # 2. Extract specific salience via Attention Rollout
            heatmap = self.explainer.generate_heatmap(attentions, inputs.input_ids, self.encoder.tokenizer)
            
            # 3. Process through Custom Multi-Label Head
            with torch.no_grad():
                probs = self.classification_head(hidden_states).squeeze(0)
            
            triggered_labels = [self.label_keys[i] for i, p in enumerate(probs) if p > 0.5]
            
            if triggered_labels:
                total_flags += len(triggered_labels)
                suspect_tokens = sorted(heatmap, key=lambda x: x["salience"], reverse=True)[:5]
                for label in triggered_labels:
                    audit_trail.append({
                        "chunk": segment_data.get("chunk_idx", 0),
                        "trigger_text": text[:200] + "...", 
                        "highlighted_tokens": suspect_tokens,
                        "violated_control": self.iso_mapping.get(label, "Unknown Control"),
                        "confidence": float(probs[self.label_keys.index(label)])
                    })
                
        return {
            "total_flags": total_flags,
            "status": "Non-Compliant" if total_flags > 0 else "Compliant",
            "audit_trail": audit_trail
        }
