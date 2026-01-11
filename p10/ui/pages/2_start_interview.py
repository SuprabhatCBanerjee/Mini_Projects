# import streamlit as st
# from api_client import start_interview
# from db import get_db

# db = get_db()

# st.title("🚀 Start Interview")

# candidates = list(db.candidates.find())
# candidate_ids = [c["_id"] for c in candidates]

# candidate_id = st.selectbox("Candidate", candidate_ids)

# if st.button("Start Interview"):
#     r = start_interview(candidate_id)

#     if r.status_code == 400:
#         st.error("Upload resume before starting interview")
#     elif r.ok:
#         st.success("Interview workflow started")
#     else:
#         st.error(r.text)

import streamlit as st
from api_client import start_interview
from db import get_db

db = get_db()

st.title("🚦 Workflow Status & Actions")

candidates = list(db.candidates.find())
candidate_ids = [c["_id"] for c in candidates]

candidate_id = st.selectbox("Candidate", candidate_ids)
candidate = db.candidates.find_one({"_id": candidate_id})

if st.button("Start Interview", key=f"start_interview_{candidate_id}"):
    r = start_interview(candidate_id)

    if r.status_code == 400:
        st.error("Upload resume before starting interview")
    elif r.ok:
        st.success("Interview workflow started")
    else:
        st.error(r.text)

st.divider()

# ---- STATUS DISPLAY ----
status = candidate.get("status", "UNKNOWN")

st.subheader("📍 Current Status")
st.code(status)

# ---- ACTION LOGIC ----
if status == "CREATED":
    st.info("Next step: Upload resume, then start interview.")

elif status == "RESUME_ANALYZED":
    st.info("Resume analyzed. Ready for technical interview.")
    if st.button("Start Interview"):
        r = start_interview(candidate_id)
        if r.ok:
            st.success("Interview started. Go to 'Submit Answer'")
            st.rerun()

elif status == "TECHNICAL_INTERVIEW":
    st.warning("Technical interview in progress.")
    st.info("Go to 'Submit Answer' and send responses.")

elif status == "BEHAVIORAL_INTERVIEW":
    st.warning("Behavioral interview in progress.")
    st.info("Continue submitting answers.")

elif status == "DECISION_READY":
    st.success("Decision ready. Go to Final Decision page.")

else:
    st.error("Unknown state")


st.subheader("🕒 Latest Activity")

latest = db.agent_outputs.find(
    {"candidate_id": candidate_id}
).sort("created_at", -1).limit(1)

for l in latest:
    st.write(f"Last agent run: {l['agent']}")
