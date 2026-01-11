import json
from datetime import datetime
from app.agents.bias_auditor import audit_bias
from app.core.mongo import (
    agent_outputs_col,
    jobs_col,
    candidates_col
)

def bias_audit_stage(state: dict) -> dict:
    candidate_id = state["candidate_id"]

    candidate = candidates_col.find_one({"_id": candidate_id})
    job = jobs_col.find_one({"_id": candidate["job_id"]})

    outputs = list(
        agent_outputs_col.find({"candidate_id": candidate_id})
    )

    audit = json.loads(
        audit_bias(job["requirements"], outputs)
    )

    agent_outputs_col.insert_one({
        "agent": "bias_auditor",
        "candidate_id": candidate_id,
        "evidence": audit,
        "created_at": datetime.utcnow()
    })

    state["bias_audited"] = True
    state["bias_flags"] = audit

    return state
