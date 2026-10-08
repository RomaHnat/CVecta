import streamlit as st
import pandas as pd
from ocr import OCR_AVAILABLE

from graph import screen, SHORTLIST_THRESHOLD

st.set_page_config(page_title="CVecta — CV Screener", page_icon="📄", layout="wide")

st.title("CVecta")
st.caption("AI-powered CV screening & job-matching — RAG + LangGraph")

st.write(
    "Paste a job description below. CVecta retrieves the most relevant CVs from the pool, "
    "scores each one against the role, and returns a ranked shortlist with reasoning."
)

job_text = st.text_area(
    "Job description",
    height=220,
    placeholder="Paste the full job description here...",
)

if st.button("Screen", type="primary"):
    if not job_text.strip():
        st.warning("Please paste a job description first.")
        st.stop()

    with st.spinner("Screening candidates..."):
        state = screen(job_text)
        if state.get("unprocessed"):
            reason = "Tesseract OCR is not installed" if not OCR_AVAILABLE else "they couldn't be read even after OCR"
            st.warning("⚠️ Not processed because " + reason + ": " + ", ".join(state["unprocessed"]))

    job = state["job"]
    ranked = state["ranked"]
    best = state["best"]

    st.subheader(f"Role: {job.title}")

    # Recommended hire (the output of the head-to-head comparison node)
    if best:
        st.success(f"**Recommended hire: {best.candidate}** — `{best.source}`\n\n{state['best_reason']}")

    rows = [
        {
            "Rank": i,
            "Candidate": r.candidate,
            "File": r.source,
            "Score": r.score,
            "Shortlisted": "✅" if r.score >= SHORTLIST_THRESHOLD else "",
            "Matched": ", ".join(r.matched),
            "Missing": ", ".join(r.missing),
            "Verdict": r.reasoning,
        }
        for i, r in enumerate(ranked, start=1)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Transparency: show what the system extracted from the job description
    with st.expander("What the system understood from the job description"):
        st.write("**Required skills:**", ", ".join(job.required_skills) or "—")
        st.write("**Nice to have:**", ", ".join(job.nice_to_have_skills) or "—")
        st.write("**Min years:**", job.min_years_experience or "—")