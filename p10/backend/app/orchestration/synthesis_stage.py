import json
from datetime import datetime
from app.agents.meta_agent import synthesize_decision
from app.core.mongo import (
    agent_outputs_col,
    jobs_col,
    candidates_col,
    final_decisions_col
)

def synthesis_stage(state: dict) -> dict:
    candidate_id = state["candidate_id"]

    candidate = candidates_col.find_one({"_id": candidate_id})
    job = jobs_col.find_one({"_id": candidate["job_id"]})

    outputs = list(
        agent_outputs_col.find(
            {"candidate_id": candidate_id, "agent": {"$ne": "meta_agent"}}
        )
    )

    bias = next(
        o["evidence"] for o in outputs if o["agent"] == "bias_auditor"
    )

    decision = json.loads(
        synthesize_decision(job["requirements"], outputs, bias)
    )

    final_decisions_col.insert_one({
        "candidate_id": candidate_id,
        "decision": decision,
        "created_at": datetime.utcnow()
    })

    candidates_col.update_one(
    {"_id": candidate_id},
    {"$set": {"status": "DECISION_READY"}}
    )

    state["decision_ready"] = True
    return state
