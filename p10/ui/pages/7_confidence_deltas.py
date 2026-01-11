import streamlit as st
from db import get_db

db = get_db()

st.title("📈 Agent Confidence Deltas")

candidates = list(db.candidates.find())
candidate_id = st.selectbox(
    "Candidate",
    [c["_id"] for c in candidates]
)

outputs = list(
    db.agent_outputs.find(
        {"candidate_id": candidate_id, "agent": "resume_intelligence"}
    ).sort("created_at", 1)
)

if len(outputs) < 2:
    st.info("Not enough data to compute deltas.")
    st.stop()

prev = outputs[-2]["evidence"]["skill_match_score"]
curr = outputs[-1]["evidence"]["skill_match_score"]

delta = curr - prev

st.metric(
    label="Resume Skill Match Change",
    value=round(curr, 2),
    delta=round(delta, 2)
)
