"""
Purpose: Parse and explicitly validate the hierarchical JSON schema for ISO compliance targets.
Goal: Produce a mathematically rigid dictionary of flattened classification labels mapped directly to their tensor weights for the PyTorch loss functions.
Scalability & Innovation: Soft-coding loss weights in a script leads to disconnected model drift. By linking the class weights and augmentation flags directly to the declarative JSON schema that holds the regulatory mapping, we enforce a single source of truth. Pydantic guarantees that if a new ISO control is added without an explicit weight, the pipeline will fail at initialization rather than silently under-performing during training.
"""
import json
import torch
from pydantic import BaseModel, RootModel, Field
from typing import Dict, List, Any

class AttributeMetadata(BaseModel):
    class_weight: float = Field(..., gt=0.0)
    augmentation_required: bool
    frameworks: List[str]

class CategorySchema(RootModel):
    root: Dict[str, AttributeMetadata]

class TargetSchema(RootModel):
    root: Dict[str, CategorySchema]

class TargetValidator:
    def __init__(self, schema_path: str = "../data/schemas/iso_targets.json"):
        self.schema_path = schema_path
        self._flattened_targets = None
        self._tensor_weights = None

    def validate_and_parse(self) -> None:
        with open(self.schema_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        validated_schema = TargetSchema.model_validate(raw_data)
        
        flat_targets = {}
        for category_name, category_data in validated_schema.root.items():
            for attribute_name, metadata in category_data.root.items():
                label_key = f"{category_name}__{attribute_name}"
                flat_targets[label_key] = metadata.model_dump()
                
        self._flattened_targets = flat_targets

    def get_loss_weights(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        if not self._flattened_targets:
            self.validate_and_parse()
            
        weights = [data["class_weight"] for data in self._flattened_targets.values()]
        return torch.tensor(weights, dtype=torch.float32, device=device)

    def get_labels(self) -> List[str]:
        if not self._flattened_targets:
            self.validate_and_parse()
        return list(self._flattened_targets.keys())
