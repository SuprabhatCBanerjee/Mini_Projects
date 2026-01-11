# import streamlit as st
# import websocket
# import json

# st.title("🎙️ Live Interview (Real-Time)")

# candidate_id = st.text_input("Candidate ID")

# if not candidate_id:
#     st.stop()

# placeholder = st.empty()

# def on_message(ws, message):
#     data = json.loads(message)
#     if data["type"] == "NEW_QUESTION":
#         placeholder.info(
#             f"[{data['stage'].upper()} QUESTION]\n\n{data['question']}"
#         )

# ws_url = f"ws://backend:8000/ws/{candidate_id}"
# ws = websocket.WebSocketApp(ws_url, on_message=on_message)

# if st.button("Connect"):
#     ws.run_forever()
# import streamlit as st
# import websocket
# import json
# import threading

# st.title("🎙️ Live Interview (Real-Time)")

# candidate_id = st.text_input("Candidate ID")

# if not candidate_id:
#     st.stop()

# placeholder = st.empty()

# def on_message(ws, message):
#     data = json.loads(message)
#     if data.get("type") == "NEW_QUESTION":
#         placeholder.info(
#             f"[{data['stage'].upper()} QUESTION]\n\n{data['question']}"
#         )

# def run_ws():
#     ws_url = f"ws://backend:8000/ws/{candidate_id}"
#     ws = websocket.WebSocketApp(ws_url, on_message=on_message)
#     ws.run_forever()

# if st.button("Connect to Live Interview"):
#     thread = threading.Thread(target=run_ws, daemon=True)
#     thread.start()
#     st.success("Connected. Waiting for questions...")

import streamlit as st
import websocket
import json
import threading
import requests
from db import get_db

# ---------------- CONFIG ----------------
API_BASE = "http://backend:8000"
WS_BASE = "ws://backend:8000"

db = get_db()

st.set_page_config(layout="wide")
st.title("🎙️ Live Interview Console")

# ---------------- STATE ----------------
if "ws_connected" not in st.session_state:
    st.session_state.ws_connected = False

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "current_stage" not in st.session_state:
    st.session_state.current_stage = None

# ---------------- INPUT ----------------
candidate_id = st.text_input("Candidate ID", key="candidate_id_input")

if not candidate_id:
    st.info("Enter candidate ID to begin.")
    st.stop()

candidate = db.candidates.find_one({"_id": candidate_id})

if not candidate:
    st.error("Invalid candidate ID.")
    st.stop()

# ---------------- STATUS PANEL ----------------
with st.expander("📍 Candidate Status", expanded=True):
    st.write("**Name:**", candidate.get("name"))
    st.write("**Job ID:**", candidate.get("job_id"))
    st.write("**Status:**", candidate.get("status", "UNKNOWN"))

st.divider()

# ---------------- WEBSOCKET HANDLER ----------------
def on_message(ws, message):
    data = json.loads(message)

    if data.get("type") == "NEW_QUESTION":
        st.session_state.current_question = data["question"]
        st.session_state.current_stage = data["stage"]
        st.experimental_rerun()


def run_ws():
    ws_url = f"{WS_BASE}/ws/{candidate_id}"
    ws = websocket.WebSocketApp(ws_url, on_message=on_message)
    ws.run_forever()


# ---------------- CONNECT BUTTON ----------------
if not st.session_state.ws_connected:
    if st.button("🔌 Connect Live Interview"):
        thread = threading.Thread(target=run_ws, daemon=True)
        thread.start()
        st.session_state.ws_connected = True
        st.success("Connected. Waiting for system events...")
        st.stop()

# ---------------- CURRENT QUESTION PANEL ----------------
st.subheader("🧠 Current Question")

if st.session_state.current_question:
    st.info(st.session_state.current_question)
    st.caption(f"Stage: {st.session_state.current_stage.upper()}")
else:
    st.warning("No active question yet.")
    st.caption("Start interview or approve a question.")

st.divider()

# ---------------- ACTIONS ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Start Interview", key=f"start_{candidate_id}"):
        r = requests.post(f"{API_BASE}/interviews/start/{candidate_id}")
        if r.ok:
            st.success("Interview started.")
        else:
            st.error(r.text)

with col2:
    if st.button("🔄 Refresh State"):
        st.experimental_rerun()

# ---------------- ANSWER INPUT ----------------
st.subheader("✍️ Submit Answer")

answer = st.text_area(
    "Candidate Answer",
    height=150,
    key=f"answer_box_{candidate_id}"
)

if st.button("📨 Submit Answer", key=f"submit_{candidate_id}"):
    if not answer.strip():
        st.warning("Answer cannot be empty.")
    else:
        r = requests.post(
            f"{API_BASE}/interviews/answer/{candidate_id}",
            json={"answer": answer}
        )
        if r.ok:
            st.success("Answer submitted. Waiting for next question...")
            st.session_state.current_question = None
            st.experimental_rerun()
        else:
            st.error(r.text)
