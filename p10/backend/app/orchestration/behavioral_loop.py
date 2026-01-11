import json
from datetime import datetime
from app.tools.behavioral_scenarios import generate_scenario
from app.agents.behavioral_agent import evaluate_behavior
from app.llm.json_utils import ensure_json

import asyncio
from app.realtime import runtime
from app.realtime.ws_manager import manager
from app.core.logger import get_logger

logger = get_logger("STAGE.BEHAVIORAL")

from app.core.mongo import (
    behavioral_interviews_col,
    agent_outputs_col,
    candidates_col
)


MAX_SCENARIOS = 3

def behavioral_interview_stage(state: dict) -> dict:
    candidate_id = state["candidate_id"]

    interview = behavioral_interviews_col.find_one(
        {"candidate_id": candidate_id}
    )

    if not interview:
        behavioral_interviews_col.insert_one({
            "candidate_id": candidate_id,
            "responses": [],
            "status": "IN_PROGRESS"
        })
        interview = behavioral_interviews_col.find_one(
            {"candidate_id": candidate_id}
        )

    if len(interview["responses"]) >= MAX_SCENARIOS:
        behavioral_interviews_col.update_one(
            {"candidate_id": candidate_id},
            {"$set": {"status": "COMPLETED"}}
        )
        state["behavioral_complete"] = True
        return state

    candidate = candidates_col.find_one({"_id": candidate_id})
    scenario = generate_scenario(role=candidate["job_id"])

    # asyncio.create_task(
    #     manager.broadcast(
    #         candidate_id,
    #         {
    #             "type": "NEW_QUESTION",
    #             "stage": "behavioral",
    #             "question": scenario
    #         }
    #     )
    # )
    if runtime.event_loop:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast(
                candidate_id,
                {
                    "type": "NEW_QUESTION",
                    "stage": "behavioral",
                    "question": scenario
                }
            ),
            runtime.event_loop
        )
        logger.info(f"[WS] Behavioral question broadcasted for {candidate_id}")

    answer = state.get("latest_answer", "No answer yet")

    evaluation = json.loads(
        evaluate_behavior(scenario, answer)
    )

    behavioral_interviews_col.update_one(
        {"candidate_id": candidate_id},
        {"$push": {
            "responses": {
                "scenario": scenario,
                "answer": answer,
                "signals": evaluation
            }
        }}
    )

    agent_outputs_col.insert_one({
        "agent": "behavioral_interviewer",
        "candidate_id": candidate_id,
        "evidence": evaluation,
        "created_at": datetime.utcnow()
    })

    candidates_col.update_one(
    {"_id": candidate_id},
    {"$set": {"status": "BEHAVIORAL_INTERVIEW"}}
    )

    return state
