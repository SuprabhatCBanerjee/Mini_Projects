from fastapi import APIRouter, HTTPException
from app.orchestration.hiring_graph import build_hiring_graph
from app.core.mongo import candidates_col, question_previews_col, technical_interviews_col
from pydantic import BaseModel
from app.orchestration.graph_runtime import graph
from app.orchestration.answer_evaluator import evaluate_answer

router = APIRouter()

from app.core.logger import get_logger

logger = get_logger("API.INTERVIEWS")

@router.post("/start/{candidate_id}")
def start(candidate_id: str):
    logger.info(f"[START] Interview start requested for {candidate_id}")
    candidate = candidates_col.find_one({"_id": candidate_id})
    logger.info(f"[END] Graph invocation completed for {candidate_id}")
    
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    if "resume_path" not in candidate:
        raise HTTPException(
            status_code=400,
            detail="Resume must be uploaded before starting interview"
        )

    graph = build_hiring_graph()
    graph.invoke({"candidate_id": candidate_id})
    return {"status": "started"}


class AnswerRequest(BaseModel):
    answer: str



# @router.post("/answer/{candidate_id}")
# def submit_answer(candidate_id: str, payload: AnswerRequest):
#     answer = payload.answer

#     # 🔽 store answer into latest question
#     technical_interviews_col.update_one(
#         {"candidate_id": candidate_id},
#         {
#             "$set": {
#                 "questions.$[q].answer": answer
#             }
#         },
#         array_filters=[{"q.answer": {"$exists": False}}]
#     )

#     return {"status": "answer_recorded"}

@router.post("/answer/{candidate_id}")
def submit_answer(candidate_id: str, payload: AnswerRequest):
    answer = payload.answer
    logger.info(f"[ANSWER] Received answer for {candidate_id}")

    # 1️⃣ Store answer in latest unanswered question
    result = technical_interviews_col.update_one(
        {"candidate_id": candidate_id},
        {
            "$set": {
                "questions.$[q].answer": answer
            }
        },
        array_filters=[{"q.answer": {"$exists": False}}]
    )

    logger.info(f"[ANSWER] Answer stored for {candidate_id}")

    evaluate_answer(candidate_id)
    # 2️⃣ RE-ENTER GRAPH to continue workflow
    logger.info(f"[GRAPH] Re-invoking graph after answer for {candidate_id}")
    # graph.invoke({"candidate_id": candidate_id})

    logger.info(f"[GRAPH] Post-answer graph invocation complete for {candidate_id}")

    return {"status": "answer_recorded"}



@router.post("/approve_question/{candidate_id}")
def approve_question(candidate_id: str):
    q = question_previews_col.find_one_and_update(
        {"candidate_id": candidate_id, "status": "PENDING"},
        {"$set": {"status": "APPROVED"}}
    )

    if not q:
        raise HTTPException(
            status_code=404,
            detail="No pending question found"
        )

    return {"approved": True, "question": q["question"]}
