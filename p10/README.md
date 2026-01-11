Below is a **clean, professional, production-grade `README.md`** you can directly drop into your repository.
It documents the **architecture, workflow, completed phases, and future roadmap** clearly — the way an experienced engineering team would.

---

#  Multi-Agent Hiring & Interview Platform

**Agentic AI · LangGraph · FastAPI · MongoDB · WebSockets · Streamlit**

A modular, event-driven **multi-agent hiring system** that conducts live technical and behavioral interviews using agent workflows, real-time updates, and human-in-the-loop controls.

This project focuses on **architecture correctness, workflow orchestration, and real-time interaction**, not just LLM calls.

---

##  Key Features

* Multi-stage hiring workflow using **LangGraph**
* Real-time interview questions via **WebSockets**
* Stateful interview loop with **MongoDB as ground truth**
* Human-in-the-loop live interview UI (**Streamlit**)
* Clean async ↔ sync boundaries
* Offline / mock-LLM mode for cost-free development
* Modular agent design (generation, evaluation, orchestration)

---

##  High-Level Architecture

```
┌─────────────┐        HTTP        ┌──────────────┐
│  Streamlit  │  ───────────────▶ │   FastAPI    │
│     UI      │                   │   Backend    │
│             │ ◀──── WebSocket ─ │              │
└─────────────┘                   └──────┬───────┘
                                          │
                                          │ invokes
                                          ▼
                                ┌────────────────────┐
                                │   LangGraph Engine  │
                                │  (Workflow Brain)  │
                                └──────┬───────┬─────┘
                                       │       │
                            Resume     │       │  Behavioral
                            Agent      │       │  Agent
                                       │
                               Technical Interview Agent
                                       │
                                       ▼
                                 MongoDB (State)
```

---

##  Project Structure

```
backend/
│
├── app/
│   ├── api/                  # FastAPI routes
│   │   ├── jobs.py
│   │   ├── candidates.py
│   │   ├── interviews.py     # start / answer endpoints
│   │   └── realtime.py       # WebSocket endpoint
│   │
│   ├── orchestration/
│   │   ├── hiring_graph.py   # LangGraph definition
│   │   ├── graph_runtime.py  # shared graph instance
│   │   ├── resume_stage.py
│   │   ├── technical_loop.py
│   │   ├── behavioral_loop.py
│   │   └── answer_evaluator.py
│   │
│   ├── llm/
│   │   ├── adapter.py        # LLM / mock adapter
│   │   └── json_utils.py     # safe JSON normalization
│   │
│   ├── realtime/
│   │   ├── runtime.py        # event loop holder
│   │   └── ws_manager.py     # WebSocket broadcast manager
│   │
│   ├── core/
│   │   ├── mongo.py
│   │   ├── logger.py
│   │   └── config.py
│   │
│   └── main.py               # FastAPI entrypoint
│
ui/
├── pages/
│   ├── 2_interviews.py
│   ├── 8_live_interview.py
│   └── ...
│
├── db.py
└── app.py
```

---

## 🔁 Interview Workflow (End-to-End)

### Phase 1: Job & Candidate Setup

* Jobs and candidates are created via REST APIs
* MongoDB stores canonical state

### Phase 2: Resume Analysis (Agent)

* Resume agent evaluates resume vs job requirements
* Results stored in `agent_outputs`
* Graph transitions automatically to technical stage

### Phase 3: Technical Interview (Agent Loop)

* First question is generated on entry
* Questions stored in `technical_interviews.questions`
* Questions pushed live via WebSocket
* Answers submitted via `/interviews/answer`
* Graph re-invoked after each answer
* Next question generated conditionally

### Phase 4: Answer Evaluation (Agent)

* Each answer evaluated independently
* Score (0–1) and depth (low / medium / high) added
* UI updates automatically (Pending → Actual)

### Phase 5: Behavioral Interview (Agent)

* Triggered after technical criteria met
* Scenario-based questions
* Same real-time loop as technical stage

### Phase 6: Final Decision (Planned)

* Aggregate scores + confidence
* Generate hiring recommendation
* Human approval step

---

## 🧠 Agent Design Principles

* **Single responsibility per agent**
* No agent writes outside its domain
* MongoDB is the **only source of truth**
* UI never infers state — only reads it
* WebSockets are **push-only**
* APIs are **event triggers**
* LangGraph controls all transitions

---

## ⚙️ Offline / Mock LLM Mode

To avoid API costs and timeouts:

```python
# app/core/config.py
LLM_ENABLED = False
```

In this mode:

* Questions, scores, and scenarios are deterministic
* No external HTTP calls
* Full system remains testable

Re-enable anytime by setting `LLM_ENABLED = True`.

---

##  Database Collections

| Collection                   | Purpose                    |
| ---------------------------- | -------------------------- |
| `jobs`                       | Job definitions            |
| `candidates`                 | Candidate profiles         |
| `agent_outputs`              | Resume analysis            |
| `technical_interviews`       | Questions, answers, scores |
| `behavioral_interviews`      | Behavioral stage           |
| *(future)* `final_decisions` | Hiring decision            |

---

##  Completed Phases

* ✔ Multi-agent LangGraph workflow
* ✔ Resume → Technical → Behavioral pipeline
* ✔ Real-time WebSocket question delivery
* ✔ Live Streamlit interview console
* ✔ Stateful interview loop
* ✔ Answer evaluation agent
* ✔ Offline / mock intelligence mode
* ✔ Robust async ↔ sync handling
* ✔ Structured logging & observability

---

##  Planned / Future Enhancements

### Intelligence

* Adaptive difficulty based on score trends
* Confidence delta tracking
* Cross-question consistency analysis

### Workflow

* Conditional graph entry (skip resume on re-invoke)
* Interview termination rules
* Retry / fallback agents

### UI / UX

* Candidate-facing interview UI
* Interview timeline visualization
* Score graphs & confidence curves

### Platform

* Auth & role-based access
* Multi-job / multi-interview support
* SaaS-ready tenant isolation

---

##  Running Locally

```bash
docker-compose up --build
```

* Backend: `http://localhost:8000`
* API Docs: `http://localhost:8000/docs`
* UI: `http://localhost:8501`

---

##  Philosophy

> This project prioritizes **correct agentic architecture** over shortcuts.
> Intelligence is modular. Workflows are explicit. State is transparent.
> The system behaves honestly — no hallucinated certainty.

---

## 📌 Status

**Architecture-complete.
Feature-extensible.
Production-ready foundation.**

---

