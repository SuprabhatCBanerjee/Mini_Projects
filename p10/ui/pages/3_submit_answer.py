import streamlit as st
from api_client import submit_answer
from db import get_db

db = get_db()

st.title("✍️ Interview – Current Question")

candidates = list(db.candidates.find())
candidate_ids = [c["_id"] for c in candidates]
candidate_id = st.selectbox("Candidate", candidate_ids)

st.divider()

# ---- DETECT CURRENT QUESTION ----
question = None
stage = None

tech = db.technical_interviews.find_one({"candidate_id": candidate_id})
beh = db.behavioral_interviews.find_one({"candidate_id": candidate_id})

if tech and tech["status"] == "IN_PROGRESS":
    if tech["questions"]:
        question = tech["questions"][-1]["question"]
        stage = "Technical Interview"

elif beh and beh["status"] == "IN_PROGRESS":
    if beh["responses"]:
        question = beh["responses"][-1]["scenario"]
        stage = "Behavioral Interview"

if question:
    st.subheader(f"🧠 {stage}")
    st.info(question)
else:
    st.success("No pending questions. Workflow may be advancing.")

st.divider()

# ---- ANSWER INPUT ----
answer = st.text_area("Your Answer", height=150)

if st.button("Submit Answer"):
    r = submit_answer(candidate_id, answer)
    if r.ok:
        st.success("Answer submitted. System advancing.")
        st.rerun()
    else:
        st.error(r.text)
