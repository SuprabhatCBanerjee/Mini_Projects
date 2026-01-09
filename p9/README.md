# The Epistemic Guardian

The Epistemic Guardian is a rule-governed epistemic evaluation system designed to analyze claims, assess their justificatory status, and expose the structure of belief rather than assert truth.

Unlike conventional AI systems that optimize for fluency or persuasion, The Epistemic Guardian enforces epistemic discipline by separating:
- language generation
- evidence retrieval
- belief justification
- confidence calibration

The system is built to **refuse unjustified certainty**, surface hidden assumptions, and make uncertainty explicit.

---

## Core Principles

- **LLMs do not decide belief**  
  Language models are used only for linguistic tasks (assumption extraction, normalization).

- **Rules govern epistemic status**  
  Deterministic logic assigns burden of proof, evaluates evidence relevance, and constrains confidence.

- **Evidence augments, never authorizes**  
  Retrieval-Augmented Generation (RAG) is used to surface potentially relevant evidence, not to infer truth.

- **Transparency over persuasion**  
  The UI exposes epistemic structure (status, assumptions, confidence ranges) without argumentation.

---

## Key Features

- Claim-level epistemic evaluation
- Burden-of-proof assignment (low / medium / high)
- Evidence–burden relevance matching
- Assumption extraction (LLM-assisted, rule-validated)
- Confidence ranges (never point estimates)
- Epistemic violation detection
- Read-only transparency UI
- Modular, auditable architecture

---

## Technology Stack

- **Python 3.11**
- **FastAPI** — API boundary and schema enforcement
- **LangGraph** — Deterministic epistemic state machine
- **LangChain** — Controlled LLM tooling
- **Hugging Face Router** — Model access via OpenAI-compatible API
- **ChromaDB** — Evidence and epistemic memory store
- **Streamlit** — Transparency UI (observer-only)

---

## Architecture Overview
---
```
User / UI
|
FastAPI (Constitutional Boundary)
|
LangGraph (Epistemic State Machine)
├─ Claim Normalization
├─ Claim Classification
├─ Assumption Extraction (LLM-assisted)
├─ RAG Evidence Lookup
├─ Burden Assignment
├─ Evidence–Burden Matching
├─ Epistemic Status Resolution
└─ Governor / Violation Detection
|
Structured Epistemic Output
```
---

## Project Structure
---
```
epistemic_guardian/
├── api/    # FastAPI entrypoint and routes
├── graph/    # LangGraph state machine and nodes
├── llm/    # LLM adapter, prompts, validators
├── rag/    # ChromaDB retrieval and ingestion
├── rules/    # Deterministic epistemic logic
├── ui/     # Streamlit transparency interface
├── run.py    # Application launcher
├── requirements.txt
└── README.md
```


---

## Running the Project
```bash
1. Install dependencies

pip install -r requirements.txt

2. Set environment variables

Create a .env file:

HF_TOKEN=your_huggingface_api_key

3. Start the API
python run.py


The API will be available at:

http://127.0.0.1:8000

4. Start the UI

In a separate terminal:

streamlit run ui/streamlit_app.py