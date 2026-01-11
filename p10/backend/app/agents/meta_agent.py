import json
from app.llm.adapter import LLMAdapter

llm = LLMAdapter()

def synthesize_decision(requirements, evidence, bias):
    prompt = f"""
You are a hiring committee synthesis generator.

JOB REQUIREMENTS:
{requirements}

EVIDENCE:
{evidence}

BIAS AUDIT:
{bias}

Return ONLY valid JSON:
- recommendation (strong_hire | hire | borderline | no_hire)
- confidence (0 to 1)
- strengths (list)
- risks (list)
- role_fit_explanation
- onboarding_focus (list)
"""

    raw = llm.call(prompt)
    return json.loads(raw)
