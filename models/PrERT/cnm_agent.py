"""
Purpose: Instantiate a Generative Agent (Contextual Neural Memory) to generate control-specific reasoning based on DeBERTa's findings.
Goal: Transform the PrERT sequence chunks and DeBERTa extracted tokens into an interpretable output JSON using a generative LLM (Mistral/Llama) and LangChain's structured output parsers.
Scalability & Innovation: The perceptual layer (DeBERTa) handles rigorous mathematical extraction and confidence matrices, while the cognitive layer (CNM Agent) generates human-readable Chain-of-Thought narratives explicitly for those mathematically triggered clauses.
"""
import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class ComplianceReasoning(BaseModel):
    thought_process: str = Field(description="A concise, single-paragraph reasoning explaining why the text violates the control.")

class CNMAgent:
    def __init__(self, llm):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=ComplianceReasoning)
        
        template = """[INST] You are an expert Privacy and Security Auditor. 
A mathematical model has already proven that the following text segment VIOLATES the specified compliance control. Your task is strictly to provide a concise, single-paragraph explanation of WHY this text segment violates the control, using the extracted tokens and broader context to form your reasoning. You must not re-evaluate the compliance status.

Triggered Text Segment:
"{text_segment}"

Broader Document Context:
"{broader_context}"

Control Violated:
{violated_control}

High-Salience Tokens (Extracted by DeBERTa):
{heatmap_tokens}

Respond entirely with a markdown JSON block explicitly matching this structure:
```json
{{
  "thought_process": "<Provide your explanation here based on the text segment>"
}}
```

Return ONLY the markdown JSON block. [/INST]
"""
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["text_segment", "broader_context", "violated_control", "heatmap_tokens"],
        )
        
        self.chain = self.prompt | self.llm | self.parser
        
    def generate_reasoning(self, heatmap_tokens: str, context: str, text_segment: str, violated_control: str) -> dict:
        try:
            return self.chain.invoke({
                "text_segment": text_segment, 
                "broader_context": context,
                "violated_control": violated_control, 
                "heatmap_tokens": heatmap_tokens
            })
        except Exception as e:
            return {
                "thought_process": f"Failed to parse LLM response: {str(e)}"
            }
