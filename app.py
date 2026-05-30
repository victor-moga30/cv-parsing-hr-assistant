import os
import pandas as pd
import streamlit as st
import plotly.express as px
from pypdf import PdfReader
from PIL import Image
import pytesseract

from src.agent import HRAssistantAgent


os.makedirs("reports", exist_ok=True)

st.set_page_config(
    page_title="CV and HR Assistant",
    page_icon="🌙",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #070314 0%, #1b0628 45%, #b85b32 100%);
    color: white;
}

h1, h2, h3, label, p, div {
    color: white;
}

.title {
    text-align: center;
    font-size: 56px;
    font-weight: 900;
    color: #ffd36e;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #ffe8bd;
    margin-bottom: 35px;
}

.stButton > button {
    background: linear-gradient(90deg, #ff7b54, #ffd36e);
    color: #12051f;
    border: none;
    border-radius: 14px;
    padding: 12px 28px;
    font-weight: 800;
    font-size: 16px;
}

.stTextArea textarea {
    background-color: #20212b;
    color: white;
    border-radius: 14px;
}

.result-card {
    background: rgba(255,255,255,0.12);
    padding: 22px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0 0 25px rgba(255,180,90,0.25);
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_agent():
    return HRAssistantAgent()


agent = load_agent()


def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""

    file_type = uploaded_file.type

    if file_type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()

    if file_type in ["image/png", "image/jpeg", "image/jpg"]:
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image).strip()

    if file_type == "text/plain":
        return uploaded_file.read().decode("utf-8").strip()

    return ""


st.markdown("<div class='title'>🌙 Midnight Sun HR Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI CV Matching • Skill Analysis • HR Recommendation</div>",
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["✨ Analiză CV individual", "📊 Batch matching"])


with tab1:
    st.markdown("## ✨ Încarcă CV și descriere job")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_cv = st.file_uploader(
            "Încarcă CV candidat",
            type=["pdf", "png", "jpg", "jpeg", "txt"]
        )

        cv_text = extract_text_from_file(uploaded_cv)

        if cv_text:
            with st.expander("📄 Text extras din CV"):
                st.write(cv_text[:4000])

    with col2:
        job_text = st.text_area(
            "Descriere job",
            height=320,
            placeholder="Lipește aici descrierea jobului..."
        )

    if st.button("Analizează potrivirea 🚀"):
        if not cv_text.strip() or not job_text.strip():
            st.warning("Încarcă un CV și completează descrierea jobului.")
        else:
            with st.spinner("AI-ul analizează CV-ul..."):
                result = agent.run(cv_text, job_text)

            st.markdown("## Rezultat analiză")

            c1, c2, c3 = st.columns(3)
            c1.metric("Scor final", f"{result['final_score']}%")
            c2.metric("Scor semantic", f"{result['semantic_score']}%")
            c3.metric("Skill score", f"{result['skill_score']}%")

            st.markdown("### 🌟 Recomandare HR")
            st.info(result["recommendation"])

            st.markdown("### ✅ Skill-uri potrivite")
            st.success(", ".join(result["matched_skills"]) or "Nu au fost detectate skill-uri comune.")

            st.markdown("### ⚠️ Skill-uri lipsă")
            st.warning(", ".join(result["missing_skills"]) or "Nu lipsesc skill-uri importante.")

            st.markdown("### 🎤 Întrebări recomandate pentru interviu")
            for q in result["interview_questions"]:
                st.write(f"- {q}")

            st.markdown("### 🚩 Red flags")
            for flag in result["red_flags"]:
                st.write(f"- {flag}")


with tab2:
    st.markdown("## 📊 Analiză batch")

    resumes_file = st.file_uploader("Încarcă processed_resumes.csv", type=["csv"])
    jobs_file = st.file_uploader("Încarcă processed_jobs.csv", type=["csv"])

    if resumes_file and jobs_file and st.button("Rulează batch matching"):
        resumes = pd.read_csv(resumes_file)
        jobs = pd.read_csv(jobs_file)

        rows = []

        with st.spinner("Se rulează analiza batch..."):
            for job_id, job in jobs.iterrows():
                job_text = str(job.get("clean_job", job.get("job_description", "")))

                for candidate_id, resume in resumes.iterrows():
                    cv_text = str(resume.get("clean_resume", resume.get("resume_text", "")))
                    result = agent.run(cv_text, job_text)

                    rows.append({
                        "job_id": job_id,
                        "candidate_id": candidate_id,
                        "final_score": result["final_score"],
                        "semantic_score": result["semantic_score"],
                        "skill_score": result["skill_score"],
                        "matched_skills": result["matched_skills"],
                        "missing_skills": result["missing_skills"],
                        "recommendation": result["recommendation"]
                    })

        results_df = pd.DataFrame(rows)
        results_df.to_csv("reports/final_results.csv", index=False)

        st.success("Analiza batch a fost finalizată.")
        st.dataframe(results_df.sort_values("final_score", ascending=False))

        fig = px.histogram(
            results_df,
            x="final_score",
            nbins=20,
            title="Distribuția scorurilor finale",
            color_discrete_sequence=["#ffd36e"]
        )
        st.plotly_chart(fig, use_container_width=True)

        top_df = results_df.sort_values("final_score", ascending=False).head(10)

        fig2 = px.bar(
            top_df,
            x="candidate_id",
            y="final_score",
            color="final_score",
            title="Top 10 candidați",
            color_continuous_scale=["#1b0628", "#ff7b54", "#ffd36e"]
        )
        st.plotly_chart(fig2, use_container_width=True)