import streamlit as st
from db import get_db
from api_client import approve_question

db = get_db()

st.title("🧪 Question Preview")

candidate_id = st.selectbox(
    "Candidate",
    [c["_id"] for c in db.candidates.find()]
)

preview = db.question_previews.find_one(
    {"candidate_id": candidate_id, "status": "PENDING"}
)

if not preview:
    st.info("No pending questions.")
    st.stop()

st.subheader("Proposed Question")
st.info(preview["question"])

if st.button("Approve"):
    approve_question(candidate_id)
    st.success("Question approved and released.")
    st.rerun()
