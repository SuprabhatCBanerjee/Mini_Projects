import streamlit as st
from api_client import create_job

st.title("🧱 Create Job")

job_id = st.text_input("Job ID")
title = st.text_input("Title")
skills = st.text_area("Required Skills (comma separated)")
experience = st.number_input("Min Experience (years)", 0, 20, 3)

if st.button("Create Job"):
    payload = {
        "job_id": job_id,
        "title": title,
        "requirements": {
            "skills": [s.strip() for s in skills.split(",") if s.strip()],
            "experience_years": experience,
        },
    }

    r = create_job(payload)
    if r.ok:
        st.success("Job created")
    else:
        st.error(r.text)
