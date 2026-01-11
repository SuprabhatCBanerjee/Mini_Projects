import streamlit as st
from db import get_db

db = get_db()

st.title("📋 Candidates")

candidates = list(db.candidates.find())

if not candidates:
    st.info("No candidates found.")
    st.stop()

for c in candidates:
    with st.expander(f"Candidate ID: {c['_id']}"):
        st.write("Job ID:", c.get("job_id"))
        st.write("Status:", c.get("status", "UNKNOWN"))
        st.write("Resume Path:", c.get("resume_path", "—"))
