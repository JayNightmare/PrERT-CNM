<p align="center">
  <a href="./Architecture-Stack.md">Architecture Stack</a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="./README.md">README</a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="./Model.md">Model Research</a>
</p>

<div align=center>

# Model Research
Research into the different models that came before and which areas to keep and what could be improved. Privacy BERT-LSTM and Polsis were good models, however their lack of tokenization caused a bottleneck in the amount of data the model could process and keep track of. The addition of the LSTM transformer allowed for short-term memory to be kept for a longer period of time, but the word embedding was not made for longer pieces of text. 

This new model purposed is called Prert-CNM (**Pr**-ivacy B-**ERT** **C**ontextual **N**eural **M**emory) increases the number of specialised parameters (increased embedding layer) and tokens (100 Input w/ 512 limit to 4096), swaps the general purpose BERT-LSTM model with the specialised mode DeBERTa v3 (with help from several other models working as helper agents) to process larger docs and hold onto more memory.

</div>

**Chapters**

* [Prert-CNM](#prert-cnm)
* [LSTM Layer](#lstm-layer)
* [Dataset](#dataset)
* [ISO Controls](#iso-controls)
* [Model Architecture](#model-architecture)
* [Output](#output)

---

# Prert-CNM

**Table of Contents**
* [Memory](#memory)
  * [Context](#context)
  * [Weightings](#weightings)
  * [Cluster Control](#cluster-control)
* [Transformer Swaps](#lstm-layer)
* [Fine-tuning LLMs](#lstm-layer)
* [Synthetic Dataset Generation](#dataset)
* [Non-Compliance Handling](#iso-controls)
* [Policy Weighting](#iso-controls)
* [Hierarchical Transformer](#hierarchical-transformer-uses)
* [Multi-label Mapping](#output)
* [Return Attributes](#returns-back)

## Memory

### Context
* Split into 3 areas
  * Broad Context
    * Flag a broad policy
    * DOES NOT apply label
  * Specialized Context
    * Checks flags using CNM
    * DOES apply label
  * Fine-Grain Context
    * Checks label against specific control
    * Flags if policy is broken
      * Will not apply anything if data is ok
* Trained on the Language of Privacy
  * Specialized Vococab
  * Train on 130k> Policies
  * Word Embeddings
    * Subword Embeddings
    * FastText
    * BERT's WordPiece
* Applying a Contextual Neural Memory (CNM)
  * Agent Framework
    * Focus
      * Data Layer *(Extraction)*
        * Segments Text into chunks
      * Application Layer *(Interface)*
        * Can ask questions about the policy
      * ML Layer *(Analysis)*
        * AI Annotation of the text
    * Design
      * NIST AI Risk Management Framework *(AI RMF)*
    
### Weightings
* Class Weighting
* Advanced Data Augmentation
* Attention Mechanism
  * Attention Weights
  * Heatmap
  * Real-time Highlighting on exact words
  * Tokens that trigger an ISO compliance failure
  * Audit Trail
 
### Cluster Control
* Vector Database
  * https://www.pinecone.io/
  * a specialized storage system designed to manage, index, and search high-dimensional vector embeddings
* Have an AI manage the cluster
  * Tasked to ensure data is processed correctly
  * Local model that lives on the server
  
## LSTM Layer
  * Swap LSTM with a pure transformer
    * RoBERTa
      * Superior Language Understanding *(Using Meta's Data)*
      * Dynamic Masking Algo
      * Variants of model
        * 12 layer w/ 125 million parameters
        * 24 layers w/ 355 million. parameters
    * DeBERTa
      * Improvement of RoBERTa and BERT *(Disentangled attention)*
      * Separates word content from its position *(processes)*
      * Replaces softmax decoder with a mask decoder
    * Fine-tuned LLM with Specialized Training/Testing
  * BERT-LSTM is a hybrid *(Note)*

## Dataset
  * Mix Datasets
  * Synthetic Dataset
    * Mix of SMS and company data
    * Based on Real-World

## ISO Controls
  * ISO Non-Compliance
    * Employ Synthetic Data Generation *(Linked to Synthetic Dataset)*
    * "Rare Event"
  * Apply Weight to policies

## Model Architecture

### Hierarchical Transformer *(Uses)*
* LLM *(Operates Through)*
  * Glass-Box
* Top Level
  * Category Layer
  * High-level ISO domain
    * Access Control
        * Linked to CNM
    * Data Retention
        * Linked to CNM
* Bottom Level
  * Attribute Layer
  * Secondary Classifier
  * Fine-grained requirements
    * Password Length
    * Encryption Standards

## Output
  * Multi-label
  * Hierarchical Attributes
    * Mapped to ISO controls

### Returns back
* Number of flags
* What data triggered a flag
* Which control was broken
