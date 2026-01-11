import streamlit as st
from pymongo import MongoClient

# --- DB CONNECTION ---
@st.cache_resource
def get_db():
    client = MongoClient("mongodb://mongo:27017")
    return client.hiring
    client = MongoClient("mongodb://mongo:27017")
    return client.hiring

db = get_db()

st.title("🗣️ Interviews")

# --- CANDIDATE SELECTION ---
candidates = list(db.candidates.find())

if not candidates:
    st.info("No candidates available.")
    st.stop()

candidate_ids = [c["_id"] for c in candidates]
candidate_id = st.selectbox("Select Candidate", candidate_ids)

st.divider()

# ======================================================
# TECHNICAL INTERVIEW SECTION
# ======================================================

st.subheader("🧠 Technical Interview")

tech = db.technical_interviews.find_one(
    {"candidate_id": candidate_id}
)

if not tech:
    st.warning("Technical interview not started.")
else:
    st.write("Status:", tech.get("status", "UNKNOWN"))
    st.write("Total Questions:", len(tech.get("questions", [])))

    for idx, q in enumerate(tech.get("questions", []), start=1):
        with st.expander(f"Question {idx}"):
            st.markdown("**Question**")
            st.write(q["question"])

            st.markdown("**Answer**")
            if "answer" in q:
                st.write("**Answer:**", q["answer"])
            else:
                st.write("_Awaiting answer..._")

            col1, col2 = st.columns(2)
            # col1.metric("Score", round(q["score"], 2))
            if "score" in q and q["score"] is not None:
                col1.metric("Score", round(q["score"], 2))
            else:
                col1.metric("Score", "Pending")

            if "depth" in q and q["depth"] is not None:    
                col2.metric("Depth", q["depth"])
            else:
                col2.metric("Depth", "Pending")

st.divider()

# ======================================================
# BEHAVIORAL INTERVIEW SECTION
# ======================================================

st.subheader("🧭 Behavioral Interview")

beh = db.behavioral_interviews.find_one(
    {"candidate_id": candidate_id}
)

if not beh:
    st.warning("Behavioral interview not started.")
else:
    st.write("Status:", beh.get("status", "UNKNOWN"))
    st.write("Scenarios Covered:", len(beh.get("responses", [])))

    for idx, r in enumerate(beh.get("responses", []), start=1):
        with st.expander(f"Scenario {idx}"):
            st.markdown("**Scenario**")
            st.write(r["scenario"])

            st.markdown("**Answer**")
            st.write(r["answer"])

            st.markdown("**Extracted Signals**")
            st.json(r["signals"])
