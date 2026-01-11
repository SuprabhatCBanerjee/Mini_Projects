import json
from app.llm.adapter import LLMAdapter

llm = LLMAdapter()

def run_resume_agent(resume_text: str, requirements: dict) -> dict:
    prompt = f"""
You are a resume analysis generator.

JOB REQUIREMENTS:
{requirements}

RESUME:
{resume_text}

Return ONLY valid JSON with:
- skill_match_score (0 to 1)
- strengths (list)
- gaps (list)
- risk_flags (list)
- summary (string)
"""

    raw = llm.call(prompt)
    return json.loads(raw)
