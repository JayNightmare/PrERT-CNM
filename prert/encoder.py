"""
Purpose: Implement the DeBERTa-based core encoder for sequence representation.
Goal: Transform raw token sequences into dense contextual embeddings while maintaining token-level mapping for downstream attention extraction.
Scalability & Innovation: Standard BERT architectures fail on long-range dependencies and relative positional encoding. By enforcing DeBERTa, we capture disentangled attention. We encapsulate the Hugging Face model to deliberately isolate the extraction of hidden states, allowing us to bypass standard classification heads and feed raw representations directly into our custom probabilistic or explicit attention engines.
"""
import torch

class PrERTEncoder:
    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        # Import inside __init__ to allow lazy loading for API servers not requiring GPU overhead
        from transformers import DebertaV2Model, DebertaV2Tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.model = DebertaV2Model.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, text: str):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state, outputs.attentions, inputs
