# Sprint 1 Implementation Report: Hierarchical Multi-Label Architecture

This report details the implementation, refactoring, and testing process taken during Month 1 to map abstract international privacy standards into verifiable, quantifiable metrics using a structured Bayesian risk engine.

## 1. Config & Data Foundation

### Hierarchical Configuration (`privacy_indicators.json`)

We pivoted from a flat metric structure to a **Hierarchical Multi-Label Classification** mapping. This enables fine-grained attributes to simultaneously map to overlapping frameworks (e.g., GDPR and NIST) by grouping them under universal categories.

```json
{
    "categories": {
        "Data_Minimization": {
            "attributes": {
                "collection_necessity_score": {
                    "frameworks": [
                        "GDPR:Article_5",
                        "NIST_Privacy_Framework:CT_DP_P"
                    ]
                }
            }
        }
    }
}
```

### Data Loading (`loader.py` & `download.py`)

To prevent the Bayesian network from collapsing under malformed constraints, we implemented rigorous schemas using `Pydantic`.

```python
class CategoryConfig(BaseModel):
    attributes: Dict[str, AttributeConfig] = Field(description="Dictionary of bottom-level attributes mapped to their framework requirements.")
```

Additionally, `data/download.py` was automated to securely fetch and locally serialize the OPP-115 policy dataset via Hugging Face `datasets`, creating the offline corpus required for Month 2 fine-tuning.

## 2. Models Layer (`privacy_bert.py`)

The perceptual layer establishes a `PrivacyFeatureExtractor` capable of wrapping Hugging Face Transformers. The extractor tokenizes incoming privacy policies and prepares the contextual output tensors required by the risk engine.

```python
# Extracting features that will eventually feed into our probabilistic engine
from models.Privacy_Bert.privacy_bert import PrivacyFeatureExtractor
extractor = PrivacyFeatureExtractor()
features = extractor.extract_features("We collect your data to improve services.")
# Tensor Geometry: [Batch Size, Hidden Dimensions]
```

## 3. Risk Engine (`bayesian_scorer.py`)

The reasoning layer replaces arbitrary risk matrices with strict probabilistic inference using `pgmpy`. The `BayesianRiskEngine` dynamically reads the hierarchical configuration and constructs a Directed Acyclic Graph (DAG) on the fly mapping `Attribute` -> `Category` -> `Framework Risk`.

```python
def build_topology_from_config(self, config: RootConfig):
    # Dynamically maps bottom-level features to higher-level frameworks
    for category_name, category_data in config.categories.items():
        for attribute_name, attribute_config in category_data.attributes.items():
            edges.append((attribute_name, category_name))
            for framework_clause in attribute_config.frameworks:
                framework_name = framework_clause.split(":")[0]
                edges.append((category_name, f"{framework_name}_Risk"))
```

## 4. Testing Process & Results (`test_pipeline.py`)

We implemented testing paradigms targeting the structural integrity of the DAG generation and the strictness of the Pydantic JSON boundaries. The testing pipeline successfully forces the system to translate complex JSON hierarchies into active probabilistic nodes.

**End-to-End Test Results (`pytest` output):**

```text
=============== test session starts ===============
platform win32 -- Python 3.12.8, pytest-8.3.4, pluggy-1.5.0 -- v:\Documents\Personal Projects\PrERT-CNM\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: v:\Documents\Personal Projects\PrERT-CNM
configfile: pytest.ini
collected 5 items

tests/test_pipeline.py::TestSyntheticRiskPipeline::test_config_validation PASSED                                            [ 20%]
tests/test_pipeline.py::TestSyntheticRiskPipeline::test_end_to_end_scoring_consistency PASSED                               [ 40%]
tests/test_pipeline.py::TestSyntheticRiskPipeline::test_bayesian_uncertainty_bounds PASSED                                  [ 60%]
tests/test_pipeline.py::TestSyntheticRiskPipeline::test_dynamic_topology_generation PASSED                                  [ 80%]
tests/test_pipeline.py::TestSyntheticRiskPipeline::test_privacy_bert_feature_extraction PASSED                              [100%]

================ 5 passed, 2 warnings in 5.42s ================
```
