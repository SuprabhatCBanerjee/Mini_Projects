# import json
# from app.tools.technical_questions import generate_question
# from app.agents.technical_agent import evaluate_answer
# from app.core.mongo import candidates_col, question_previews_col
# from app.core.mongo import (
#     technical_interviews_col,
#     agent_outputs_col
# )


# from app.realtime.ws_manager import manager
# import asyncio

# from datetime import datetime

# MAX_QUESTIONS = 5

# def technical_interview_stage(state: dict) -> dict:
#     candidate_id = state["candidate_id"]

#     interview = technical_interviews_col.find_one(
#         {"candidate_id": candidate_id}
#     )

#     if not interview:
#         technical_interviews_col.insert_one({
#             "candidate_id": candidate_id,
#             "questions": [],
#             "status": "IN_PROGRESS"
#         })
#         interview = technical_interviews_col.find_one(
#             {"candidate_id": candidate_id}
#         )

#     if len(interview["questions"]) >= MAX_QUESTIONS:
#         technical_interviews_col.update_one(
#             {"candidate_id": candidate_id},
#             {"$set": {"status": "COMPLETED"}}
#         )
#         state["technical_complete"] = True
#         return state

#     # question = generate_question(
#     #     skill="Python",
#     #     difficulty="medium"
#     # )
#     preview = {
#         "candidate_id": candidate_id,
#         "stage": "technical",
#         "question": generate_question("Python", "medium"),
#         "status": "PENDING",
#         "created_at": datetime.utcnow()
#     }

    

#     asyncio.create_task(
#         manager.broadcast(
#             candidate_id,
#             {
#                 "type": "NEW_QUESTION",
#                 "stage": "technical",
#                 "question": preview
#             }
#         )
#     )

#     # In real system: answer comes from UI
#     answer = state.get("latest_answer", "No answer yet")

#     evaluation = json.loads(
#         evaluate_answer(preview["question"], answer)
#     )

#     technical_interviews_col.update_one(
#         {"candidate_id": candidate_id},
#         {"$push": {
#             "questions": {
#                 "question": preview["question"],
#                 "answer": answer,
#                 "score": evaluation["score"],
#                 "depth": evaluation["depth_detected"]
#             }
#         }}
#     )

#     agent_outputs_col.insert_one({
#         "agent": "technical_interviewer",
#         "candidate_id": candidate_id,
#         "evidence": evaluation,
#         "created_at": datetime.utcnow()
#     })

#     candidates_col.update_one(
#     {"_id": candidate_id},
#     {"$set": {"status": "TECHNICAL_INTERVIEW"}}
#     )

#     question_previews_col.insert_one(preview)
#     state["waiting_for_approval"] = True
#     return state


import asyncio
from datetime import datetime
from app.core.mongo import technical_interviews_col
from app.realtime.ws_manager import manager
from app.tools.technical_questions import generate_question

from app.realtime import runtime

from app.core.logger import get_logger
logger = get_logger("STAGE.TECHNICAL")

def technical_interview_stage(state: dict) -> dict:
    candidate_id = state["candidate_id"]
    logger.info(f"[TECH] Entered technical stage for {candidate_id}")

    # 1️⃣ Ensure document exists
    ti = technical_interviews_col.find_one(
        {"candidate_id": candidate_id}
    )

    if not ti:
        logger.info(f"[TECH] Creating technical interview shell for {candidate_id}")
        technical_interviews_col.insert_one({
            "candidate_id": candidate_id,
            "questions": [],
            "status": "IN_PROGRESS",
            "created_at": datetime.utcnow()
        })
        ti = technical_interviews_col.find_one(
            {"candidate_id": candidate_id}
        )

    # 2️⃣ If NO questions yet → generate FIRST question
    if len(ti["questions"]) == 0:
        logger.info(f"[TECH] Generating FIRST question for {candidate_id}")
        question = generate_question(
            skill="Python",
            difficulty="medium"
        )

        technical_interviews_col.update_one(
            {"candidate_id": candidate_id},
            {
                "$push": {
                    "questions": {
                        "question": question,
                        "created_at": datetime.utcnow()
                    }
                }
            }
        )
        logger.info(f"[TECH] Question stored in MongoDB for {candidate_id}")

        # 3️⃣ Broadcast immediately
        # asyncio.create_task(
        #     manager.broadcast(
        #         candidate_id,
        #         {
        #             "type": "NEW_QUESTION",
        #             "stage": "technical",
        #             "question": question
        #         }
        #     )
        # )
        
        if runtime.event_loop:
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(
                    candidate_id,
                    {
                        "type": "NEW_QUESTION",
                        "stage": "technical",
                        "question": question
                    }
                ),
                runtime.event_loop
            )
            logger.info(f"[WS] Question broadcasted for {candidate_id}")
      

    # 4️⃣ Otherwise, wait for answer
    return state
