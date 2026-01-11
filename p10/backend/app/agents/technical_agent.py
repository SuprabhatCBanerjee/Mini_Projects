import json
from app.llm.adapter import LLMAdapter

llm = LLMAdapter()

def evaluate_answer(question: str, answer: str) -> dict:
    prompt = f"""
You are a technical evaluation generator.

QUESTION:
{question}

ANSWER:
{answer}

Return ONLY valid JSON:
- score (0 to 1)
- depth_detected (shallow | medium | deep)
- follow_up_needed (true/false)
- notes
"""

    raw = llm.call(prompt)
    return json.loads(raw)
