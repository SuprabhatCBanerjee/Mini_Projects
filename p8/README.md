#  Multi-Agent Devil's Advocate

A **local, multi-agent deliberation engine** that forces structured
disagreement before decisions are made.

This system is **not a chatbot**.\
It is a **decision-support framework** designed to surface trade-offs,
risks, human impact, and feasibility by simulating opposing
institutional perspectives.

Built with: - LangGraph - LangChain - LM Studio + Mistral 7B (local
inference) - FastAPI - Streamlit

------------------------------------------------------------------------

## ✨ What This Project Is

A **deliberative AI system** where independent agents analyze the same
question from conflicting incentives:

-   Optimizer --- Growth, speed, upside\
-   Risk Guardian --- Failure modes, worst cases\
-   Human Impact --- Ethics, social consequences\
-   Cost & Feasibility --- Budget, complexity, realism\
-   Moderator --- Synthesizes disagreement (no final decision)

The system intentionally avoids producing a single "answer".

------------------------------------------------------------------------

## 🚫 What This Project Is NOT

-   Not a chatbot
-   Not a recommender system
-   Not a single-prompt wrapper
-   Not an automation engine

This tool exists to **slow decisions down**, not speed them up.

------------------------------------------------------------------------

## 🏗️ Architecture Overview

Frontend (Streamlit)\
→ Backend (FastAPI)\
→ LangGraph (multi-agent reasoning)\
→ Local LLM (Mistral 7B via LM Studio)

Key principles: - Independent reasoning - Explicit disagreement - No
forced consensus - Fully local execution

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   Python 3.10+
-   LangChain
-   LangGraph
-   FastAPI
-   Streamlit
-   LM Studio
-   Mistral 7B (GGUF)

------------------------------------------------------------------------

## 💻 Hardware Requirements

Minimum recommended: - GPU: RTX 1650 (4 GB VRAM) - RAM: 16 GB - Disk:
\~8 GB free

Runs fully offline after setup.

------------------------------------------------------------------------

## 📦 Installation

``` bash
git clone https://github.com/yourusername/multi-agent-devils-advocate.git
cd multi-agent-devils-advocate
python -m venv venv
```

Activate environment and install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🤖 Local Model Setup (LM Studio)

1.  Install LM Studio: https://lmstudio.ai\
2.  Download: mistral-7b-instruct.Q4_K\_M.gguf\
3.  Settings:
    -   Context: 2048
    -   Temperature: 0.6
    -   GPU layers: 20--28
4.  Enable OpenAI-compatible API

Server:

    http://localhost:1234/v1

------------------------------------------------------------------------

## ▶️ Running the System

``` bash
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

Open:

    http://localhost:8501

------------------------------------------------------------------------

## 🧭 Modes

**Fast Mode** - Optimizer - Risk - Moderator

**Deep Mode** - All agents - Slower, more thorough analysis

------------------------------------------------------------------------

## ⏱️ Observability

Each agent logs execution time in backend logs.

------------------------------------------------------------------------

## 🧠 Why This Matters

Most AI systems collapse uncertainty. This one exposes it.

It is designed to: - reduce overconfidence - reveal hidden assumptions -
make trade-offs explicit

------------------------------------------------------------------------

## 📈 Future Work

-   Parallel agents
-   Streaming results
-   Decision memory
-   Disagreement scoring
-   Hybrid local/cloud moderation

------------------------------------------------------------------------

## ⚠️ Ethical Use

This system does not make decisions. Humans remain responsible.

------------------------------------------------------------------------

## 📜 License

MIT License
