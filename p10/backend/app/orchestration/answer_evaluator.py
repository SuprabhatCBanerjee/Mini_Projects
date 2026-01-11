from app.core.logger import get_logger
from app.llm.adapter import LLMAdapter
from app.core.mongo import technical_interviews_col

logger = get_logger("AGENT.EVALUATOR")

llm = LLMAdapter()

def evaluate_answer(candidate_id: str):
    ti = technical_interviews_col.find_one({"candidate_id": candidate_id})
    if not ti or not ti["questions"]:
        return

    # get last answered question
    for q in reversed(ti["questions"]):
        if "answer" in q and "score" not in q:
            question = q["question"]
            answer = q["answer"]
            break
    else:
        return  # nothing to evaluate

    logger.info(f"[EVAL] Evaluating answer for {candidate_id}")

    prompt = f"""
You are a technical interviewer.

Question:
{question}

Answer:
{answer}

Return JSON ONLY with:
- score: number between 0 and 1
- depth: low | medium | high
"""

    response = llm.call(prompt)

    try:
        result = eval(response)  # safe enough for now (we harden later)
    except Exception:
        logger.warning("[EVAL] Failed to parse LLM response")
        return

    technical_interviews_col.update_one(
        {"candidate_id": candidate_id, "questions.answer": answer},
        {
            "$set": {
                "questions.$.score": result["score"],
                "questions.$.depth": result["depth"]
            }
        }
    )

    logger.info(
        f"[EVAL] Score={result['score']} Depth={result['depth']} stored"
    )
