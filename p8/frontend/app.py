import streamlit as st
import requests

st.set_page_config(layout="wide")
st.title("🧠 Multi-Agent Devil’s Advocate")
mode = st.radio(
    "Mode",
    ["Fast", "Deep"],
    horizontal=True
)
question = st.text_area(
    "Enter a proposal, policy, or decision:",
    height=120
)

if st.button("Analyze"):
    with st.spinner("Agents are arguing..."):
        res = requests.post(
            "http://localhost:8000/debate",
            json={"question": question}
        ).json()

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.subheader("🚀 Optimizer")
        st.write(res["optimizer_view"])

    with col2:
        st.subheader("⚠️ Risk Guardian")
        st.write(res["risk_view"])

    with col3:
        st.subheader("🧑‍🤝‍🧑 Human Impact")
        st.write(res["human_view"])

    with col4:
        st.subheader("💰 Cost & Feasibility")
        st.write(res["cost_view"])

    st.divider()
    st.subheader("🧩 Moderator Synthesis")
    st.write(res["synthesis"])
