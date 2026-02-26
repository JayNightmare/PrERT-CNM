"""
Purpose: Orchestrate the PrERT-CNM pipeline from ingestion through encoding to final hierarchical ISO control mapping.
Goal: Provide a unified interface that returns multi-label compliance evaluations, complete with audit trails, flag counts, and trigger mappings.
Scalability & Innovation: This is not a simple linear pipeline. It represents a stateful compliance machine. By chaining the Context Memory Bank with DeBERTa embeddings and our custom Attention Explainer, we construct a deterministic, traceable evaluation graph that outputs structured risk data, completely divorcing ourselves from opaque end-to-end classification paradigms.
"""
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
from pathlib import Path
from typing import Any
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline
from dotenv import load_dotenv

from .ingestion import DocumentIngestor
from .encoder import PrERTEncoder
from .attention import AttentionExplainer
from .cnm_agent import CNMAgent

class PrERTPipeline:
    def __init__(self):
        self.ingestor = DocumentIngestor()
        self.encoder = PrERTEncoder()
        self.explainer = AttentionExplainer(class_weights={"iso_27001": 1.5, "gdpr": 1.2})
        
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
            
        self.agent = CNMAgent(self.llm) if self.llm else None
        
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

    @staticmethod
    def _hard_violation_rules() -> list[tuple[str, str]]:
        return [
            (
                "Sells user personal data to third parties",
                r"(?:\b(?:sell|sold|selling|commercialize|monetiz\w*)\b.{0,60}\b(?:personal data|user data|private data|customer data|data)\b|\b(?:personal data|user data|private data|customer data|data)\b.{0,60}\b(?:sell|sold|selling|commercialize|monetiz\w*)\b)",
            ),
            (
                "No encryption statement",
                r"(?:\b(?:do\s+not|does\s+not|no|without|not)\b.{0,12}\b(?:encrypt(?:ion|ed)?|cryptograph(?:y|ic)|cipher(?:s|ed)?)\b|\b(?:unencrypted|plaintext)\b.{0,25}\b(?:data|database|storage|records|traffic)\b)",
            ),
            (
                "Indefinite data retention",
                r"(?:\bretain(?:ed|s|ing)?\b.{0,35}\b(?:indefinitely|forever|without deletion)\b|\b(?:indefinitely|forever)\b.{0,35}\bretain(?:ed|s|ing)?\b)",
            ),
            (
                "Forced opt-in without consent",
                r"(?:\b(?:automatically\s+opt(?:ed)?\s+in|opt\s+you\s+in|opted\s+in\s+by\s+default)\b.{0,90}\b(?:without|no)\b.{0,45}\b(?:consent|explicit consent)\b)",
            ),
        ]

    @staticmethod
    def _extract_trigger_sentence(text: str, start_idx: int, end_idx: int, max_chars: int = 420) -> str:
        if not text:
            return ""

        delimiters = [".", "!", "?", "\n"]
        left = max([text.rfind(d, 0, start_idx) for d in delimiters])
        right_candidates = [text.find(d, end_idx) for d in delimiters]
        right_candidates = [idx for idx in right_candidates if idx != -1]

        left = 0 if left == -1 else left + 1
        right = min(right_candidates) if right_candidates else len(text)

        excerpt = " ".join(text[left:right].strip().split())
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars].rstrip() + "..."
        return excerpt

    @staticmethod
    def _build_rule_based_heatmap(trigger_text: str, matched_text: str) -> list[dict[str, Any]]:
        stop_words = {
            "the", "and", "or", "a", "an", "this", "that", "is", "are", "to", "for", "of",
            "in", "on", "with", "by", "we", "you", "your", "our", "their", "be", "as", "all"
        }

        matched_terms = {
            term.lower()
            for term in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", matched_text or "")
            if term.lower() not in stop_words
        }

        tokens = []
        seen = set()
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", trigger_text or ""):
            lowered = token.lower()
            if lowered in stop_words or lowered in seen:
                continue
            seen.add(lowered)

            is_match = lowered in matched_terms
            tokens.append({
                "token": token,
                "weight": 1.0 if is_match else 0.45,
                "category": "red" if is_match else "green",
            })

            if len(tokens) >= 12:
                break

        return tokens

    @classmethod
    def _extract_hard_violation_matches(cls, text: str, policy_id: int = -1) -> list[dict[str, Any]]:
        raw_text = text or ""
        extracted = []

        for description, pattern in cls._hard_violation_rules():
            for match in re.finditer(pattern, raw_text, flags=re.IGNORECASE):
                trigger_text = cls._extract_trigger_sentence(raw_text, match.start(), match.end())
                if not trigger_text:
                    continue

                matched_text = raw_text[match.start():match.end()]
                extracted.append({
                    "policy_id": int(policy_id),
                    "description": description,
                    "triggered_text": trigger_text,
                    "cause_heatmap": cls._build_rule_based_heatmap(trigger_text, matched_text),
                })

        deduped = []
        seen = set()
        for item in extracted:
            key = (item["policy_id"], item["description"], item["triggered_text"].lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @classmethod
    def _collect_hard_violation_matches(cls, all_chunks: list, full_text: str) -> list[dict[str, Any]]:
        collected = []

        for chunk in all_chunks or []:
            metadata = chunk.get("metadata", {})
            policy_id = metadata.get("policy_id", -1)
            segment_text = chunk.get("segment", "")
            collected.extend(cls._extract_hard_violation_matches(segment_text, policy_id=policy_id))

        if not collected:
            collected = cls._extract_hard_violation_matches(full_text, policy_id=-1)

        deduped = []
        seen = set()
        for item in collected:
            key = (item["policy_id"], item["description"], item["triggered_text"].lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @classmethod
    def _detect_hard_violation_phrases(cls, text: str) -> list[str]:
        matches = cls._extract_hard_violation_matches(text, policy_id=-1)
        ordered_descriptions = []
        seen = set()
        for match in matches:
            description = match["description"]
            if description not in seen:
                seen.add(description)
                ordered_descriptions.append(description)
        return ordered_descriptions

    @staticmethod
    def _derive_trigger_excerpt(text: str, suspect_tokens: list[dict[str, Any]], max_chars: int = 420) -> str:
        raw_text = text or ""
        if not raw_text.strip():
            return ""

        candidate_tokens = [
            (token.get("token") or "").lower()
            for token in suspect_tokens
            if token.get("category") in {"red", "green"} and token.get("token")
        ]

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[\.!\?])\s+|\n+", raw_text)
            if sentence.strip()
        ]

        if not sentences:
            excerpt = " ".join(raw_text.split())
            return (excerpt[:max_chars].rstrip() + "...") if len(excerpt) > max_chars else excerpt

        if not candidate_tokens:
            excerpt = sentences[0]
            return (excerpt[:max_chars].rstrip() + "...") if len(excerpt) > max_chars else excerpt

        best_sentence = sentences[0]
        best_score = -1
        for sentence in sentences:
            lowered = sentence.lower()
            score = sum(1 for token in candidate_tokens if token and token in lowered)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        if len(best_sentence) > max_chars:
            return best_sentence[:max_chars].rstrip() + "..."
        return best_sentence

    def process_document(self, content: bytes | str, is_pdf: bool = False, status_updater=None) -> dict:
        session_id = str(uuid.uuid4())

        if isinstance(content, bytes):
            try:
                full_text = content.decode("utf-8", errors="ignore")
            except Exception:
                full_text = ""
        else:
            full_text = str(content)
        
        if status_updater: status_updater("Parsing document into semantic chunks using ChromaDB...")
        memory = self.ingestor.process_document(content, is_pdf, session_id)
        print("Completed Ingestion")
            
        audit_trail = []
        total_flags = 0
        violated_controls = []
        
        all_chunks = memory.retrieve_context(session_id=session_id)
        
        for idx, segment_data in enumerate(all_chunks):
            chunk_idx = segment_data["metadata"].get("chunk_idx", 0)
            policy_id = segment_data["metadata"].get("policy_id", 0)
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
                
                entailment_idx = self.encoder.model.config.label2id.get("entailment", self.encoder.model.config.label2id.get("ENTAILMENT", 1))
                
                entailment_logit = logits[0, entailment_idx]
                probs = F.softmax(logits, dim=-1)
                prob = probs[0, entailment_idx].item()
                
                is_compliant = prob <= 0.5
                thought_process = "Segment is mathematically compliant."
                suspect_tokens = []
                salience_tokens_str = ""
                reflection_attempts = 0
                
                if not is_compliant:
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
                            if t["weight"] > 0.7:
                                t["category"] = "red"
                            elif t["weight"] > 0.3:
                                t["category"] = "green"
                            else:
                                t["category"] = "blue"
                                    
                    salience_tokens_str = ", ".join([f"{t['token']} (Weight: {t['weight']:.2f})" for t in suspect_tokens if t["category"] in ["red", "green"]])
                    
                    if self.agent:
                        attempts = 0
                        max_attempts = 3
                        test_passed = False
                        feedback = None
                        
                        while attempts < max_attempts and not test_passed:
                            attempts += 1
                            if status_updater: status_updater(f"Evaluating contextual compliance with Mistral-7B (Attempt {attempts}/{max_attempts}) for chunk {idx+1}/{len(all_chunks)}...")
                            
                            analysis = self.agent.generate_reasoning(salience_tokens_str, broader_context, text, violated_control, feedback)
                            review_is_compliant = analysis.get("is_compliant", False)
                            thought_process = analysis.get("thought_process", "No reasoning provided.")
                            
                            if status_updater: status_updater(f"Testing Review Agent's logic (Attempt {attempts}/{max_attempts}) for chunk {idx+1}/{len(all_chunks)}...")
                            test_result = self.agent.generate_test(text, broader_context, violated_control, review_is_compliant, thought_process)
                            
                            if test_result.get("is_correct", False):
                                test_passed = True
                            else:
                                feedback = test_result.get("feedback", "Reasoning was deemed incorrect by the Test Layer.")
                                if status_updater and attempts < max_attempts:
                                    status_updater(f"Testing layer found a false positive/true negative and is trying again with attempt {attempts + 1}...")
                        reflection_attempts = attempts

                        is_compliant = review_is_compliant
                    else:
                        thought_process = "Generative Agent offline. Relying on mathematical Zero-Shot entailment."
                        
                    if not is_compliant:
                        total_flags += 1
                        violated_controls.append({
                            "control": violated_control,
                            "confidence": float(prob)
                        })

                trigger_excerpt = text if not is_compliant else text
                if not is_compliant:
                    trigger_excerpt = self._derive_trigger_excerpt(text, suspect_tokens)

                audit_trail.append({
                    "policy_id": policy_id,
                    "control_name": violated_control,
                    "triggered_status": not is_compliant,
                    "triggered_text": trigger_excerpt,
                    "cnm_reason": thought_process,
                    "cause_heatmap": suspect_tokens,
                    "attempts": reflection_attempts if self.agent else (1 if not is_compliant else 0)
                })

        hard_violation_matches = self._collect_hard_violation_matches(all_chunks, full_text)
        if hard_violation_matches:
            if status_updater:
                status_updater("Absolute-violation safety layer triggered due to explicit policy language.")

            for match in hard_violation_matches:
                total_flags += 1
                violated_controls.append({
                    "control": "Absolute Violation Safety Layer",
                    "confidence": 1.0
                })
                audit_trail.append({
                    "policy_id": match["policy_id"],
                    "control_name": "Absolute Violation Safety Layer",
                    "triggered_status": True,
                    "triggered_text": match["triggered_text"],
                    "cnm_reason": "Detected explicit high-risk policy language: " + match["description"],
                    "cause_heatmap": match["cause_heatmap"],
                    "attempts": 0
                })
                
        return {
            "status": "Non-Compliant" if total_flags > 0 else "Compliant",
            "total_flags": total_flags,
            "violated_controls": violated_controls,
            "audit_trail": audit_trail
        }

    def run(self, content: bytes | str, is_pdf: bool = False, status_updater=None):
        return self.process_document(content, is_pdf, status_updater)
