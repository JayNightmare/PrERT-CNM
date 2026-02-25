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

class NLIClassificationHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, 1)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features[:, 0, :]
        x = x.to(self.dense.weight.dtype)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class PrERTPipeline:
    def __init__(self):
        self.ingestor = DocumentIngestor()
        self.encoder = PrERTEncoder()
        
        self.explainer = AttentionExplainer(class_weights={"iso_27001": 1.5, "gdpr": 1.2})
        
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline as hf_pipeline
        from dotenv import load_dotenv
        
        load_dotenv()
        
        print("Initializing Generative LLM for Contextual Neural Memory...")
        try:
            generator = hf_pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.2", 
                device=0 if torch.cuda.is_available() else -1
            )
            
            if hasattr(generator, "model") and hasattr(generator.model, "generation_config"):
                generator.model.generation_config.max_length = None
                
            self.llm = HuggingFacePipeline(
                pipeline=generator,
                pipeline_kwargs={
                    "max_new_tokens": 512,
                    "return_full_text": False,
                    "pad_token_id": 2
                }
            )
        except Exception as e:
            print(f"Warning: Failed to load Generative LLM: {str(e)}")
            self.llm = None
            
        from .cnm_agent import CNMAgent
        self.agent = CNMAgent(self.llm) if self.llm else None
        
        import json
        from pathlib import Path
        schema_path = Path(__file__).parent.parent / "data" / "schemas" / "iso_targets.json"
        
        self.iso_mapping = {}
        self.iso_hypotheses = {}
        with open(schema_path, "r") as f:
            targets = json.load(f)
            
        for category, attributes in targets.items():
            for attr_name, attr_data in attributes.items():
                framework_str = " | ".join(attr_data["frameworks"])
                concept = attr_name.replace("_", " ").lower()
                hypothesis = f"This policy violates privacy standards regarding {concept}."
                self.iso_mapping[attr_name] = framework_str
                self.iso_hypotheses[attr_name] = hypothesis
                
        if "Missing_Encryption_At_Rest" in self.iso_hypotheses:
            self.iso_hypotheses["Missing_Encryption_At_Rest"] = "This policy lacks adequate encryption or cryptography standards for data at rest and in transit."
            
        self.label_keys = list(self.iso_mapping.keys())
        self.classification_head = NLIClassificationHead(
            hidden_size=self.encoder.model.config.hidden_size
        ).to(self.encoder.device).to(self.encoder.model.dtype)

    def process_document(self, content: bytes | str, is_pdf: bool = False) -> dict:
        import uuid
        session_id = str(uuid.uuid4())
        
        memory = self.ingestor.process_document(content, is_pdf, session_id)
        print("Completed Ingestion")
            
        audit_trail = []
        total_flags = 0
        violated_controls = []
        
        all_chunks = memory.retrieve_context(session_id=session_id)
        
        for segment_data in all_chunks:
            chunk_idx = segment_data["metadata"].get("chunk_idx", 0)
            text = segment_data["segment"]
            
            broader_context_pieces = []
            if chunk_idx > 0 and chunk_idx - 1 < len(all_chunks):
                broader_context_pieces.append(all_chunks[chunk_idx - 1]["segment"])
            broader_context_pieces.append(text)
            if chunk_idx + 1 < len(all_chunks):
                broader_context_pieces.append(all_chunks[chunk_idx + 1]["segment"])
            broader_context = " ... ".join(broader_context_pieces)
            
            for label in self.label_keys:
                hypothesis = self.iso_hypotheses[label]
                violated_control = self.iso_mapping.get(label, "Unknown Control")
                
                self.encoder.model.zero_grad()
                self.classification_head.zero_grad()
                hidden_states, attentions, inputs = self.encoder.encode(text, hypothesis, require_grad=True)
                print("Completed Encoding")
                
                last_layer_attn = attentions[-1]
                last_layer_attn.retain_grad()
                
                entailment_logit = self.classification_head(hidden_states).squeeze(0)
                prob = torch.sigmoid(entailment_logit).item()
                
                heatmap = self.explainer.extract_heatmap(
                    entailment_logit, last_layer_attn, inputs.input_ids, self.encoder.tokenizer
                )
                print("Completed Attention Seeking")
                
                suspect_tokens = sorted(heatmap, key=lambda x: x["weight"], reverse=True)[:15]
                
                if suspect_tokens:
                    max_weight = suspect_tokens[0]["weight"]
                    for t in suspect_tokens:
                        t["weight"] = (t["weight"] / max_weight) if max_weight > 0 else 0.0
                        
                        if prob > 0.5:
                            if t["weight"] > 0.7:
                                t["category"] = "red"
                            elif t["weight"] > 0.3:
                                t["category"] = "green"
                            else:
                                t["category"] = "blue"
                        else:
                            if t["weight"] > 0.5:
                                t["category"] = "green"
                            else:
                                t["category"] = "blue"
                                
                if prob > 0.5:
                    total_flags += 1
                    is_compliant = False
                    
                    salience_tokens_str = ", ".join([f"{t['token']} (Weight: {t['weight']:.2f})" for t in suspect_tokens if t["category"] in ["red", "green"]])
                    
                    if self.agent:
                        analysis = self.agent.generate_reasoning(salience_tokens_str, broader_context, text, violated_control)
                        thought_process = analysis.get("thought_process", "No reasoning provided.")
                    else:
                        thought_process = "Generative Agent offline. Cannot compute reasoning."
                        
                    violated_controls.append({
                        "control": violated_control,
                        "confidence": float(prob)
                    })
                else:
                    is_compliant = True
                    thought_process = "DeBERTa mathematically asserts this segment does NOT entail a violation of the specified control."
                        
                audit_trail.append({
                    "control_name": violated_control,
                    "triggered_status": not is_compliant,
                    "triggered_text": text,
                    "cnm_reason": thought_process,
                    "cause_heatmap": suspect_tokens
                })
                
        return {
            "status": "Non-Compliant" if total_flags > 0 else "Compliant",
            "total_flags": total_flags,
            "violated_controls": violated_controls,
            "audit_trail": audit_trail
        }
