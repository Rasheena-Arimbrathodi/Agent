# System Architecture

## Overview
EvidenceFlow separates **retrieval**, **tool execution**, and **response synthesis** so every claim can be audited against source documents.

## Components
1. **Retriever (`evidenceflow.retriever`)**: Tokenizes documents, removes stopwords, and computes a weighted combination of BM25 and TF-IDF cosine similarity.
2. **Tools (`evidenceflow.tools`)**:
   - `search_docs`: Top-k retrieval returning snippet, score, and document ID.
   - `read_doc`: Retrieves the complete text and metadata of a specific document ID.
   - `calculate`: Evaluates mathematical expressions safely via Python's `ast` module without `eval()`.
3. **Agent (`evidenceflow.agent`)**: Inspects the query, invokes tools in sequence, tracks execution traces, and formats cited answers.
4. **Evaluator (`evidenceflow.evaluator`)**: Runs benchmark queries against expected citations and answer criteria.
