import streamlit as st
from db import get_db

db = get_db()

st.title("⚖️ Bias & Fairness Audit")

candidate_ids = [c["_id"] for c in db.candidates.find()]
candidate_id = st.selectbox("Select Candidate", candidate_ids)

audit = db.agent_outputs.find_one({
    "candidate_id": candidate_id,
    "agent": "bias_auditor"
})

if not audit:
    st.info("Bias audit not yet completed.")
    st.stop()

e = audit["evidence"]

st.metric("Bias Detected", str(e["bias_detected"]))
st.metric("Severity", e["severity"])

st.subheader("Issues")
for issue in e.get("issues", []):
    st.write("•", issue)

st.subheader("Explanation")
st.write(e["explanation"])
