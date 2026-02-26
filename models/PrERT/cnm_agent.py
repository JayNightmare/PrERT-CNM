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
    is_compliant: bool = Field(description="True if the text is actually compliant (a false positive), False if it is a true violation of the control.")
    thought_process: str = Field(description="A concise, single-paragraph reasoning explaining whether the text violates the control and why.")

class TestValidation(BaseModel):
    is_correct: bool = Field(description="True if the Review Agent's compliance evaluation is logically sound and accurate, False otherwise.")
    feedback: str = Field(description="If not correct, provide specific feedback on what the Review Agent got wrong. If correct, return 'None'.")

class CNMAgent:
    def __init__(self, llm):
        self.llm = llm
        self.review_parser = JsonOutputParser(pydantic_object=ComplianceReasoning)
        self.test_parser = JsonOutputParser(pydantic_object=TestValidation)
        
        review_template = """[INST] You are an expert Privacy and Security Auditor. 
A mathematical model has flagged the following text segment for potentially violating the specified compliance control. 
Your task is to re-evaluate this flag. Does this text truly violate the control, or is it a false positive?
Provide a concise explanation of your reasoning based on the extracted tokens and context, and definitively state whether it is compliant.

{feedback_section}
Triggered Text Segment:
"{text_segment}"

Broader Document Context:
"{broader_context}"

Control in Question:
{violated_control}

High-Salience Tokens (Extracted by Model):
{heatmap_tokens}

Respond entirely with a markdown JSON block explicitly matching this structure:
```json
{{
  "is_compliant": <true or false>,
  "thought_process": "<Provide your explanation here>"
}}
```

Return ONLY the markdown JSON block. [/INST]
"""
        
        test_template = """[INST] You are a highly critical QA Engineer for a Privacy Auditing System.
A Review Agent has just evaluated a potentially violating text segment against a compliance control.
Your task is to verify if the Review Agent's final decision is logically correct based on the provided text, context, and control.

Triggered Text Segment:
"{text_segment}"

Broader Document Context:
"{broader_context}"

Control in Question:
{violated_control}

Review Agent's Conclusion:
Is Compliant: {is_compliant}
Reasoning: {thought_process}

If the Review Agent is correct, set is_correct to true.
If the Review Agent made a mistake (e.g., missed an absolute violation or incorrectly flagged compliant text), set is_correct to false and provide feedback for the Review Agent to try again.

Respond entirely with a markdown JSON block explicitly matching this structure:
```json
{{
  "is_correct": <true or false>,
  "feedback": "<Specific instructions on what is wrong, or 'None' if correct>"
}}
```

Return ONLY the markdown JSON block. [/INST]
"""
        
        self.review_prompt = PromptTemplate(
            template=review_template,
            input_variables=["text_segment", "broader_context", "violated_control", "heatmap_tokens", "feedback_section"],
        )
        self.review_chain = self.review_prompt | self.llm | self.review_parser

        self.test_prompt = PromptTemplate(
            template=test_template,
            input_variables=["text_segment", "broader_context", "violated_control", "is_compliant", "thought_process"],
        )
        self.test_chain = self.test_prompt | self.llm | self.test_parser
        
    def generate_reasoning(self, heatmap_tokens: str, context: str, text_segment: str, violated_control: str, feedback: str = None) -> dict:
        feedback_section = f"Previous Feedback to Address:\n{feedback}\n" if feedback else ""
        try:
            return self.review_chain.invoke({
                "text_segment": text_segment, 
                "broader_context": context,
                "violated_control": violated_control, 
                "heatmap_tokens": heatmap_tokens,
                "feedback_section": feedback_section
            })
        except Exception as e:
            return {
                "is_compliant": False, # Default to failure on parse error
                "thought_process": f"Failed to parse LLM response: {str(e)}"
            }

    def generate_test(self, text_segment: str, context: str, violated_control: str, is_compliant: bool, thought_process: str) -> dict:
        try:
            return self.test_chain.invoke({
                "text_segment": text_segment,
                "broader_context": context,
                "violated_control": violated_control,
                "is_compliant": is_compliant,
                "thought_process": thought_process
            })
        except Exception as e:
            return {
                "is_correct": True, # Fail open to avoid infinite retry loops on parse err
                "feedback": f"Test parser failed: {str(e)}"
            }

    def evaluate_reasoning(self, text_segment: str, context: str, violated_control: str, is_compliant: bool, thought_process: str) -> dict:
        return self.generate_test(text_segment, context, violated_control, is_compliant, thought_process)
