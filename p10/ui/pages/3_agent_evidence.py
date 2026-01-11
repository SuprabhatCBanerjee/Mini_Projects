import streamlit as st
from db import get_db

db = get_db()

st.title("🧠 Agent Evidence Viewer")

candidate_ids = [c["_id"] for c in db.candidates.find()]
candidate_id = st.selectbox("Select Candidate", candidate_ids)

outputs = list(db.agent_outputs.find(
    {"candidate_id": candidate_id}
).sort("created_at", 1))

if not outputs:
    st.warning("No agent evidence available.")
    st.stop()

for o in outputs:
    st.subheader(f"Agent: {o['agent']}")
    st.json(o["evidence"])
    st.caption(f"Timestamp: {o['created_at']}")
