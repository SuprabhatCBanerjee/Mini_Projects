import streamlit as st
from api_client import create_candidate, upload_resume
from db import get_db

db = get_db()

st.title("👤 Create Candidate")

jobs = list(db.jobs.find())
job_ids = [j["_id"] for j in jobs]

name = st.text_input("Candidate Name")
job_id = st.selectbox("Job ID", job_ids)
resume = st.file_uploader("Upload Resume (PDF/DOCX)")

if st.button("Create Candidate"):
    r = create_candidate(name, job_id)
    if r.ok:
        cid = r.json()["candidate_id"]
        st.success(f"Candidate created: {cid}")

        if resume:
            upload_resume(cid, resume)
            st.success("Resume uploaded")
    else:
        st.error(r.text)
