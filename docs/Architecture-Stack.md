<p align="center">
  <a href="./Architecture-Stack.md">Architecture Stack</a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="./README.md">README</a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="./Model.md">Model Research</a>
</p>

<div align=center>

# Architecture Stack

</div>

**Chapters**

- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)

---

# Tech Stack

Proposed text stack for building the model architecture

**Table of Contents**

- [Architecture Stack](#architecture-stack)
- [Tech Stack](#tech-stack)
  - [Deep Learning Framework](#deep-learning-framework)
  - [Programming Language](#programming-language)
  - [Secure Hardware/Execution](#secure-hardwareexecution)
  - [Data Processing and Tokenisation](#data-processing-and-tokenisation)
  - [Text Segmentation](#text-segmentation)
  - [Model Architecture Specifics](#model-architecture-specifics)
    - [ISO Documents:](#iso-documents)
    - [Hig Acc NLU:](#hig-acc-nlu)
  - [Database Management](#database-management)
- [How it works](#how-it-works)
  - [Front-End](#front-end)
    - [Query Module](#query-module)
    - [Class Comparison Engine](#class-comparison-engine)
    - [Interpretability Dashboard](#interpretability-dashboard)
  - [Back-End](#back-end)
    - [Data Layer Management](#data-layer-management)
    - [Machine Learning Layer Management](#machine-learning-layer-management)
    - [Risk and Security Management](#risk-and-security-management)

</div>

## Deep Learning Framework

- Multi-Transformer Arch _(Contextual Embedding)_
- Convolutional Neural Networks (CNN) and Contextual Neural Memory (CNM) _(ReLU)_

## Programming Language

- Python (Backend) _(Standard for LLMs)_
    - PyTorch _(Deep Learning Framework)_
    - HuggingFace Transformer
    - vLLM _(Interface Engine)_
- React TypeScript (Frontend)

## Secure Hardware/Execution

To establish TEEs, the hardware would require one of the following:

- Intel SGX
- AMD
- Arm TrustZone

## Data Processing and Tokenisation

- NLTK _(Text Processing and Tokenisation)_
- HuggingFace Dataset _(Apache Arrow)_
    - Memory mapped storage _(stored in secure folders)_

## Text Segmentation

- GraphSeg _(Unsupervised Algorithm)_

## Model Architecture Specifics

### ISO Documents:

- BigBird
    - 4096 tokens
    - Ideal for understanding long texts
    - 128 million parameters

### Hig Acc NLU:

- DeBERTa-v3 _(Outpreforms BERT, RoBERTa and DeBERTa versions)_
    - Base Model :~: 184 million _(parameters)_
    - Large :~: 435 million
    - XLarge :~: 750 million
    - XXLarge :~: 1.5 billion

## Database Management

- ChromaDB/Pinecone/Milvus/pgvector _(Vector)_
    - Zilliz Cloud _(Milvus Serverless option - reduces maintenance)_
- SQLite/PostgreSQL _(Relational)_
- LangChain/LlamaIndex _(Orchestrator)_

<div align=center>

# How it works

How the model will operate, align itself, and component expansion

</div>

**Table of Contents**

- [Architecture Stack](#architecture-stack)
- [Tech Stack](#tech-stack)
  - [Deep Learning Framework](#deep-learning-framework)
  - [Programming Language](#programming-language)
  - [Secure Hardware/Execution](#secure-hardwareexecution)
  - [Data Processing and Tokenisation](#data-processing-and-tokenisation)
  - [Text Segmentation](#text-segmentation)
  - [Model Architecture Specifics](#model-architecture-specifics)
    - [ISO Documents:](#iso-documents)
    - [Hig Acc NLU:](#hig-acc-nlu)
  - [Database Management](#database-management)
- [How it works](#how-it-works)
  - [Front-End](#front-end)
    - [Query Module](#query-module)
    - [Class Comparison Engine](#class-comparison-engine)
    - [Interpretability Dashboard](#interpretability-dashboard)
  - [Back-End](#back-end)
    - [Data Layer Management](#data-layer-management)
    - [Machine Learning Layer Management](#machine-learning-layer-management)
    - [Risk and Security Management](#risk-and-security-management)

## Front-End

### Query Module

- Structure Queries _(Incoming)_
- Natural Language Queries _(Incoming)_
- User Interaction _(Gateway)_

### Class Comparison Engine

- Matches ISO control attributes against user query _(Model Prediction Classes)_
- Returns relevant policy segments _(User Specific)_

### Interpretability Dashboard

- Attention Heatmap _(Display)_
- Highlights flagged words/tokens _(Display)_
- Compliance vs Non-compliance _(Number x Report)_

## Back-End

### Data Layer Management

- Policy Extraction
    - Crawler
    - Document Parser
    - Removing Irrelevant metadata
- List Aggregation and Handling
    - Lists are merged into paragraphs
    - Long lists are split but include context statements and labels
- Semantic Segmentation
    - Algo to break doc into segments
    - Ensure context is kept with cliques of related sentences

### Machine Learning Layer Management

- Domain-Specific Embeddings
    - Subword Embeddings
    - Train on a massive number of specific security policies
- Hierarchical Multi-Label Classification
    - 2 stage model: Top and Bottom
    - Predict High-Level ISO domain
    - Predict Low-Level classifiers for fine-grain attributes
- Class Imbalance Handling
    - Data Augmentation
    - Class Weighting

### Risk and Security Management

- NIST AI Risk Management Framework
    - Development and Deployment Life Cycle _(Governed)_
    - Must incorporate trustworthiness, transparency, and risk mitigation _(Continuously)_
- Trusted Execution Environments
    - Hardware-enforced isolation
- Threat Modeling
    - Threat Model Testing _(Before Deployment)_
        - OWASP
        - Microsoft Threat Modeling Tool
