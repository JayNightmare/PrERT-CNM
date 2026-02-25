"""
Purpose: Orchestrate the PrERT-CNM pipeline from ingestion through encoding to final hierarchical ISO control mapping.
Goal: Provide a unified interface that returns multi-label compliance evaluations, complete with audit trails, flag counts, and trigger mappings.
Scalability & Innovation: This is not a simple linear pipeline. It represents a stateful compliance machine. By chaining the Context Memory Bank with DeBERTa embeddings and our custom Attention Explainer, we construct a deterministic, traceable evaluation graph that outputs structured risk data, completely divorcing ourselves from opaque end-to-end classification paradigms.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ingestion import DocumentIngestor
from .encoder import PrERTEncoder
from .attention import AttentionExplainer

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
        schema_path = Path(__file__).parent.parent.parent / "data" / "schemas" / "iso_targets.json"
        
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

    def process_document(self, content: bytes | str, is_pdf: bool = False, status_updater=None) -> dict:
        import uuid
        session_id = str(uuid.uuid4())
        
        if status_updater: status_updater("Parsing document into semantic chunks using ChromaDB...")
        memory = self.ingestor.process_document(content, is_pdf, session_id)
        print("Completed Ingestion")
            
        audit_trail = []
        total_flags = 0
        violated_controls = []
        
        all_chunks = memory.retrieve_context(session_id=session_id)
        
        for idx, segment_data in enumerate(all_chunks):
            chunk_idx = segment_data["metadata"].get("chunk_idx", 0)
            text = segment_data["segment"]
            
            broader_context_pieces = []
            if chunk_idx > 0 and chunk_idx - 1 < len(all_chunks):
                broader_context_pieces.append(all_chunks[chunk_idx - 1]["segment"])
            broader_context_pieces.append(text)
            if chunk_idx + 1 < len(all_chunks):
                broader_context_pieces.append(all_chunks[chunk_idx + 1]["segment"])
            broader_context = " ... ".join(broader_context_pieces)
            
            for label_idx, label in enumerate(self.label_keys):
                hypothesis = self.iso_hypotheses[label]
                violated_control = self.iso_mapping.get(label, "Unknown Control")
                
                if status_updater: status_updater(f"Encoding chunk {idx+1}/{len(all_chunks)} for control: {violated_control}...")
                self.encoder.model.zero_grad()
                logits, attentions, inputs = self.encoder.encode(text, hypothesis, require_grad=True)
                print("Completed Encoding")
                
                last_layer_attn = attentions[-1]
                last_layer_attn.retain_grad()
                
                # NLI Model logit indices: 0: contradiction, 1: entailment, 2: neutral
                entailment_logit = logits[0, 1]
                probs = F.softmax(logits, dim=-1)
                prob = probs[0, 1].item()
                
                if status_updater: status_updater(f"Extracting transformer attention rollout for chunk {idx+1}/{len(all_chunks)}...")
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
                                
                salience_tokens_str = ", ".join([f"{t['token']} (Weight: {t['weight']:.2f})" for t in suspect_tokens if t["category"] in ["red", "green"]])
                
                if self.agent:
                    if status_updater: status_updater(f"Evaluating contextual compliance with Mistral-7B for chunk {idx+1}/{len(all_chunks)}...")
                    analysis = self.agent.generate_reasoning(salience_tokens_str, broader_context, text, violated_control)
                    is_compliant = analysis.get("is_compliant", prob <= 0.5)
                    thought_process = analysis.get("thought_process", "No reasoning provided.")
                else:
                    is_compliant = prob <= 0.5
                    thought_process = "Generative Agent offline. Relying on mathematical Zero-Shot."

                if not is_compliant:
                    total_flags += 1
                    violated_controls.append({
                        "control": violated_control,
                        "confidence": float(prob)
                    })
                        
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
