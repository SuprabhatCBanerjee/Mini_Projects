import json
from app.llm.adapter import LLMAdapter
from app.llm.json_utils import ensure_json


llm = LLMAdapter()

def evaluate_behavior(scenario: str, answer: str) -> dict:
    prompt = f"""
You are a behavioral signal extractor.

SCENARIO:
{scenario}

ANSWER:
{answer}

Return ONLY valid JSON with:
- ownership (0 to 1)
- clarity (0 to 1)
- ethical_awareness (0 to 1)
- consistency (0 to 1)
- risk_flags (list)
"""

    raw = llm.call(prompt)
    return ensure_json(raw)

