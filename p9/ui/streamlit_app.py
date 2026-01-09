import streamlit as st
import requests
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/evaluate"

st.set_page_config(
    page_title="Epistemic Guardian",
    layout="centered"
)

st.title("🛡️ The Epistemic Guardian")
st.caption("Epistemic transparency interface (read-only)")

st.divider()

# Claim Input
claim = st.text_area(
    "Enter a single claim to evaluate",
    placeholder="e.g. AI will replace most programmers in 5 years",
    height=80
)

evaluate = st.button("Evaluate Claim")

# Evaluation Call
if evaluate and claim.strip():
    with st.spinner("Evaluating epistemic status..."):
        response = requests.post(
            API_URL,
            json={"claim": claim}
        )

    if response.status_code != 200:
        st.error("Guardian failed to evaluate the claim.")
    else:
        data = response.json()

        st.divider()

        
        # Claim Summary
        st.subheader("Claim")
        st.write(data["normalized_claim"])

        
        # Epistemic Status
        st.subheader("Epistemic Status")
        st.code(data["epistemic_status"])

        
        # Confidence Range
        st.subheader("Confidence Range")

        low, high = data["confidence_range"]

        fig, ax = plt.subplots()
        ax.barh(["Confidence"], [high - low], left=low)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_yticks([])

        st.pyplot(fig)

        st.caption(f"Range: {low:.2f} – {high:.2f}")

        
        # Assumptions
        st.subheader("Detected Assumptions")
        if data["assumptions"]:
            for a in data["assumptions"]:
                st.write("•", a)
        else:
            st.write("No explicit assumptions detected.")

        
        # Evidence Snippets (RAG)
        st.subheader("Retrieved Evidence (RAG)")
        st.caption("Evidence is evaluated for relevance vs burden, not volume.")
        if data["evidence_snippets"]:
            for i, e in enumerate(data["evidence_snippets"], 1):
                st.write(f"{i}. {e}")
        else:
            st.write("No relevant evidence retrieved.")

        
        # Epistemic Violations
        if data["violations"]:
            st.subheader("Epistemic Violations")
            for v in data["violations"]:
                st.error(v)
        else:
            st.subheader("Epistemic Violations")
            st.success("No epistemic violations detected.")

