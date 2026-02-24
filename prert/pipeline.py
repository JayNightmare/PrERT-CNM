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
        # Ensure input dtype aligns with layers to avoid Half/Float mismatch
        x = x.to(self.dense.weight.dtype)
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
        ).to(self.encoder.device).to(self.encoder.model.dtype)

    def process_document(self, content: bytes | str, is_pdf: bool = False) -> dict:
        import uuid
        session_id = str(uuid.uuid4())
        
        if is_pdf:
            memory = self.ingestor.process_pdf(content, session_id)
        else:
            memory = self.ingestor.process_text(str(content), session_id)
            
        audit_trail = []
        total_flags = 0
        
        for segment_data in memory.retrieve_context(session_id=session_id):
            text = segment_data["segment"]
            
            # 1. Encode into context-aware representations (Ensure grad is enabled for backward pass)
            self.encoder.model.zero_grad()
            self.classification_head.zero_grad()
            hidden_states, attentions, inputs = self.encoder.encode(text, require_grad=True)
            
            # Hook the last layer's attention for gradient extraction
            last_layer_attn = attentions[-1]
            last_layer_attn.retain_grad()
            
            # 3. Process through Custom Multi-Label Head
            probs = self.classification_head(hidden_states).squeeze(0)
            
            triggered_labels = [self.label_keys[i] for i, p in enumerate(probs) if p > 0.5]
            
            if triggered_labels:
                total_flags += len(triggered_labels)
                
                for label in triggered_labels:
                    class_idx = self.label_keys.index(label)
                    target_logit = probs[class_idx]
                    
                    # Generate class-specific heatmap via Gradient-Weighted Attention
                    heatmap = self.explainer.generate_class_heatmap(
                        target_logit, last_layer_attn, inputs.input_ids, self.encoder.tokenizer
                    )
                    
                    # Embedded full top 10 token heatmap to audit trail based on "weight" extraction
                    suspect_tokens = sorted(heatmap, key=lambda x: x["weight"], reverse=True)[:10]
                    
                    # Normalize weights to range [0.2 - 1.0] for visible UI heatmap rendering
                    if suspect_tokens:
                        max_weight = suspect_tokens[0]["weight"]
                        for t in suspect_tokens:
                            t["weight"] = min(1.0, max(0.2, t["weight"] / max_weight)) if max_weight > 0 else 0.2
                            
                    audit_trail.append({
                        "chunk": segment_data.get("chunk_idx", 0),
                        "trigger_text": text[:200] + "...", 
                        "highlighted_tokens": suspect_tokens,
                        "violated_control": self.iso_mapping.get(label, "Unknown Control"),
                        "confidence": float(probs[class_idx])
                    })
                
        return {
            "total_flags": total_flags,
            "status": "Non-Compliant" if total_flags > 0 else "Compliant",
            "audit_trail": audit_trail
        }
