<div align=center>

# Tech Stack
Proposed text stack for building the model architecture

</div>

## Deep Learning Framework

* Multi-Transformer Arch *(Contextual Embedding)*
* Convolutional Neural Networks (CNN) and Contextual Neural Memory (CNM) *(ReLU)*


## Programming Language

* Python (Backend) *(Standard for LLMs)*
    * PyTorch *(Deep Learning Framework)*
    * HuggingFace Transformer 
    * vLLM *(Interface Engine)*
* React TypeScript (Frontend)


## Secure Hardware/Execution
To establish TEEs, the hardware would require one of the following:
* Intel SGX
* AMD
* Arm TrustZone


## Data Processing and Tokenisation

* NLTK *(Text Processing and Tokenisation)*
* HuggingFace Dataset *(Apache Arrow)*
    * Memory mapped storage *(stored in secure folders)*


## Text Sementation

* GraphSeg *(Unsupervised Algorithm)*


## Model Architecture Specifics

### ISO Documents:
* Clinical-Longformer
    * 4096 tokens
    * More Clinical Tuned
* BigBird
    * 4096 tokens
    * Ideal for understanding long texts
    * 128 million parameters

### Hig Acc NLU:
* DeBERTa-v3 *(Outpreforms BERT, RoBERTa and DeBERTa versions)*
    * Base Model :~: 184 million *(parameters)*
    * Large :~: 435 million
    * XLarge :~: 750 million
    * XXLarge :~: 1.5 billion


## Database Management

* Pinecone/Milvus/pgvector *(Vector)*
    * Zilliz Cloud *(Milvus Serverless option - reduces maintenance)*
* SQLite/PostgreSQL *(Relational)*
* LangChain/LlamaIndex *(Orchestrator)*

<div align=center>

# How it works
How the model will operate, align itself, and component expansion

</div>

## Front-End

* Query Module
  * Structure Queries *(Incoming)*
  * Natural Language Queries *(Incoming)*
  * User Interaction *(Gateway)*
* Class Comparison Engine
  * Matches ISO control attributes against user query *(Model Prediction Classes)*
  * Returns relevant policy segments *(User Specific)*
* Interpretability Dashboard
  * Attention Heatmap *(Display)*
  * Highlights flagged words/tokens *(Display)*
  * Compliance vs Non-compliance *(Number x Report)*


## Back-End

* Data Layer Management
  * Policy Extraction
    * Crawler
    * Document Parser
    * Removing Irrelevant metadata
  * List Aggregation and Handling
    * Lists are merged into paragraphs
    * Long lists are split but include context statements and labels
  * Semantic Segmentation
    * Algo to break doc into segments
    * Ensure context is kept with cliques of related sentences
* Machine Learning Layer Management
  * Domain-Specific Embeddings
    * Subword Embeddings
    * Train on a massive number of specific security policies
  * Hierarchical Multi-Label Classification
    * 2 stage model: Top and Bottom
    * Predict High-Level ISO domain
    * Predict Low-Level classifiers for fine-grain attributes
  * Class Imbalance Handling
    * Data Augmentation
    * Class Weighting
* Risk and Security Management
  * NIST AI Risk Management Framework
    * Development and Deployment Life Cycle *(Governed)*
    * Must incorporate trustworthiness, transparency, and risk mitigation *(Continuously)*
  * Trusted Execution Environments
    * Hardware-enforced isolation
  * Threat Modeling
    * Threat Model Testing *(Before Deployment)*
      * OWASP
      * Microsoft Threat Modeling Tool
