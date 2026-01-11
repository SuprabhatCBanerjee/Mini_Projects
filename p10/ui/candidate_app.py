import streamlit as st
import requests

API_BASE = "http://backend:8000"

st.title("🎤 Candidate Interview Portal")

candidate_id = st.text_input("Enter your Candidate ID")

if not candidate_id:
    st.stop()

# Fetch current question (read-only)
r = requests.get(f"{API_BASE}/candidates/{candidate_id}")
if not r.ok:
    st.error("Invalid candidate ID")
    st.stop()

st.divider()

st.subheader("Current Question")
st.info("Please answer the question shown by the interviewer.")

answer = st.text_area("Your Answer", height=200)

if st.button("Submit"):
    r = requests.post(
        f"{API_BASE}/interviews/answer/{candidate_id}",
        json={"answer": answer}
    )
    if r.ok:
        st.success("Answer submitted. Please wait.")
    else:
        st.error(r.text)
