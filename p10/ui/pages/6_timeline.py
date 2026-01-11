import streamlit as st
from db import get_db

db = get_db()

st.title("🕒 Candidate Timeline")

candidates = list(db.candidates.find())
candidate_ids = [c["_id"] for c in candidates]
candidate_id = st.selectbox("Candidate", candidate_ids)

st.divider()

events = list(
    db.agent_outputs.find({"candidate_id": candidate_id})
    .sort("created_at", 1)
)

if not events:
    st.info("No activity yet.")
    st.stop()

for e in events:
    with st.expander(
        f"{e['created_at']} — {e['agent']}"
    ):
        st.json(e["evidence"])
