# from app.tools.resume_parser import parse_resume
# from app.agents.resume_agent import run_resume_agent
# from app.core.mongo import candidates_col, jobs_col, agent_outputs_col

# import json
# from datetime import datetime

# def resume_stage(state: dict) -> dict:
#     candidate_id = state["candidate_id"]

#     candidate = candidates_col.find_one({"_id": candidate_id})
#     job = jobs_col.find_one({"_id": candidate["job_id"]})

#     resume_text = parse_resume(candidate["resume_path"])
#     output = run_resume_agent(resume_text, job["requirements"])

#     parsed = json.loads(output)

#     agent_outputs_col.insert_one({
#         "agent": "resume_intelligence",
#         "candidate_id": candidate_id,
#         "evidence": parsed,
#         "created_at": datetime.utcnow()
#     })

#     state["resume_complete"] = True
#     return state

from app.tools.resume_parser import parse_resume
from app.agents.resume_agent import run_resume_agent
from app.core.mongo import candidates_col, jobs_col, agent_outputs_col
from datetime import datetime
import json


# def resume_stage(state: dict) -> dict:
#     candidate_id = state["candidate_id"]

#     candidate = candidates_col.find_one({"_id": candidate_id})
#     if not candidate:
#         raise ValueError(f"Candidate {candidate_id} not found")

#     # 🔒 PRECONDITION CHECK
#     resume_path = candidate.get("resume_path")
#     if not resume_path:
#         # Do NOT crash the system
#         # Mark state and stop progression
#         state["resume_missing"] = True
#         return state

#     job = jobs_col.find_one({"_id": candidate["job_id"]})

#     resume_text = parse_resume(resume_path)
#     output = run_resume_agent(resume_text, job["requirements"])

#     agent_outputs_col.insert_one({
#         "agent": "resume_intelligence",
#         "candidate_id": candidate_id,
#         "evidence": output,
#         "created_at": datetime.utcnow()
#     })

#     state["resume_complete"] = True
#     return state

from app.core.logger import get_logger

logger = get_logger("STAGE.RESUME")

def resume_stage(state: dict) -> dict:
    candidate_id = state["candidate_id"]
    logger.info(f"[RESUME] Entered resume stage for {candidate_id}")    

    candidate = candidates_col.find_one({"_id": candidate_id})
    if not candidate:
        raise ValueError(f"Candidate {candidate_id} not found")

    resume_path = candidate.get("resume_path")
    if not resume_path:
        state["resume_missing"] = True
        return state

    job_id = candidate.get("job_id")
    job = jobs_col.find_one({"_id": job_id})

    if not job:
        # 🔒 Do not crash graph
        state["job_missing"] = True
        state["error"] = f"Job {job_id} not found"
        return state

    resume_text = parse_resume(resume_path)
    output = run_resume_agent(resume_text, job["requirements"])

    agent_outputs_col.insert_one({
        "agent": "resume_intelligence",
        "candidate_id": candidate_id,
        "evidence": output,
        "created_at": datetime.utcnow()
    })

    candidates_col.update_one(
    {"_id": candidate_id},
    {"$set": {"status": "RESUME_ANALYZED"}}
    )

    

    state["resume_complete"] = True
    logger.info(f"[RESUME] Resume analysis complete for {candidate_id}")
    return state
