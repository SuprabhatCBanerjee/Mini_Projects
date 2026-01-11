import json
from app.llm.adapter import LLMAdapter

llm = LLMAdapter()

def audit_bias(requirements: dict, agent_outputs: list) -> dict:
    prompt = f"""
You are a bias and fairness audit generator.

JOB REQUIREMENTS:
{requirements}

AGENT EVIDENCE:
{agent_outputs}

Analyze for:
- irrelevant signal weighting
- pedigree proxies
- inconsistency
- ignored uncertainty

Return ONLY valid JSON:
- bias_detected (true/false)
- issues (list)
- severity (low | medium | high)
- explanation
"""

    raw = llm.call(prompt)
    return json.loads(raw)
