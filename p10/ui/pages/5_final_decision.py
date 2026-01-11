import streamlit as st
from db import get_db

db = get_db()

st.title("Final Decision & Human Review")

candidate_ids = [c["_id"] for c in db.candidates.find()]
candidate_id = st.selectbox("Select Candidate", candidate_ids)

decision = db.final_decisions.find_one(
    {"candidate_id": candidate_id}
)

if decision:
    st.subheader("🤖 AI Committee Recommendation")
    st.json(decision["decision"])
else:
    st.warning("No AI decision available yet.")
    st.stop()

st.divider()
st.subheader("👤 Human Review")

decision_choice = st.selectbox(
    "Your Decision",
    ["approved", "rejected", "deferred"]
)

notes = st.text_area("Reviewer Notes")

if st.button("Submit Review"):
    db.human_reviews.insert_one({
        "candidate_id": candidate_id,
        "decision": decision_choice,
        "notes": notes
    })
    st.success("Human review recorded.")
