import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agent import HRAssistantAgent
from src.document_parser import extract_text_from_uploaded_file
from src.skill_extraction import extract_skills


os.makedirs("reports", exist_ok=True)

st.set_page_config(
    page_title="CV Parsing and HR Assistant",
    page_icon="HR",
    layout="wide",
)


st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(135deg, #0f1020 0%, #2b0f22 45%, #6f1d1b 100%);
    color: white;
}

h1, h2, h3, h4, label, p, div {
    color: white;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 900;
    color: #f7d08a;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #fbe7c6;
    margin-bottom: 35px;
}

.stButton > button {
    background: linear-gradient(90deg, #9d0208, #f48c06);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-weight: 800;
    font-size: 16px;
}

.stTextArea textarea {
    background-color: #1f1f2e;
    color: white;
    border-radius: 12px;
}

.result-card {
    background: rgba(255,255,255,0.10);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.22);
    box-shadow: 0 0 20px rgba(244,140,6,0.18);
    margin-top: 12px;
}

.small-muted {
    color: #fbe7c6;
    font-size: 14px;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_agent():
    return HRAssistantAgent()


agent = load_agent()


def render_candidate_details(row):
    st.markdown(
        f"""
        <div class="result-card">
            <h3>Rank {int(row["rank"])}: {row["candidate_name"]}</h3>
            <p><b>File:</b> {row["file_name"]}</p>
            <p><b>Verdict:</b> {row["verdict"]}</p>
            <p><b>Final score:</b> {row["final_score"]}%</p>
            <p><b>Semantic score:</b> {row["semantic_score"]}%</p>
            <p><b>Skill score:</b> {row["skill_score"]}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_results_dataframe(results):
    rows = []

    for index, item in enumerate(results):
        result = item["result"]
        info = result["applicant_info"]

        rows.append({
            "candidate_index": index + 1,
            "file_name": item["file_name"],
            "candidate_name": info.get("name", "Unknown"),
            "email": info.get("email", "Unknown"),
            "phone": info.get("phone", "Unknown"),
            "linkedin": info.get("linkedin", "Unknown"),
            "github": info.get("github", "Unknown"),
            "text_length": info.get("text_length", 0),
            "final_score": result["final_score"],
            "semantic_score": result["semantic_score"],
            "baseline_score": result["baseline_score"],
            "skill_score": result["skill_score"],
            "tfidf_score": result["tfidf_score"],
            "matched_skills_count": len(result["matched_skills"]),
            "missing_skills_count": len(result["missing_skills"]),
            "matched_skills": ", ".join(result["matched_skills"]),
            "missing_skills": ", ".join(result["missing_skills"]),
            "cv_skills": ", ".join(result["cv_skills"]),
            "job_skills": ", ".join(result["job_skills"]),
            "recommendation": result["recommendation"],
            "interview_questions": " | ".join(result["interview_questions"]),
            "red_flags": " | ".join(result["red_flags"]),
            "agent_steps": " -> ".join(result.get("agent_steps", [])),
            "verdict": result["verdict"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


st.markdown("<div class='main-title'>CV Parsing and HR Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI agent for CV parsing, matching, ranking, recommendations and HR analytics</div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "CV matching",
    "Dashboard",
    "Model metrics",
    "Agent explanation",
])

if "results_df" not in st.session_state:
    st.session_state["results_df"] = None

if "raw_results" not in st.session_state:
    st.session_state["raw_results"] = None


with tab1:
    st.markdown("## Input")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_cvs = st.file_uploader(
            "Upload one or more CV files",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            accept_multiple_files=True,
        )

        show_extracted_text = st.checkbox("Show extracted text and detected skills", value=False)

        if uploaded_cvs:
            st.info(f"Uploaded CV files: {len(uploaded_cvs)}")

    with col2:
        job_text = st.text_area(
            "Job description",
            height=300,
            placeholder="Paste the job description here...",
        )

        if job_text.strip():
            st.markdown("Detected job skills:")
            st.write(extract_skills(job_text))

    analyze_button = st.button("Analyze and generate ranking")

    if analyze_button:
        if not uploaded_cvs:
            st.warning("Upload at least one CV.")
        elif not job_text.strip():
            st.warning("Complete the job description.")
        else:
            raw_results = []

            with st.spinner("Extracting CV text and running the AI agent..."):
                for uploaded_file in uploaded_cvs:
                    cv_text = extract_text_from_uploaded_file(uploaded_file)

                    if show_extracted_text:
                        st.markdown(f"### Extracted text from {uploaded_file.name}")
                        st.write(cv_text[:3000])

                        st.markdown("Detected CV skills:")
                        st.write(extract_skills(cv_text))

                    if not cv_text.strip():
                        st.warning(f"Could not extract text from file: {uploaded_file.name}")
                        continue

                    result = agent.run(cv_text, job_text)

                    raw_results.append({
                        "file_name": uploaded_file.name,
                        "cv_text": cv_text,
                        "result": result,
                    })

            if not raw_results:
                st.error("No CV could be analyzed.")
            else:
                results_df = build_results_dataframe(raw_results)

                st.session_state["results_df"] = results_df
                st.session_state["raw_results"] = raw_results

                results_df.to_csv("reports/hr_dashboard_results.csv", index=False)

                st.success("Analysis completed.")
                st.markdown("## Candidate ranking")

                visible_columns = [
                    "rank",
                    "candidate_name",
                    "file_name",
                    "final_score",
                    "semantic_score",
                    "skill_score",
                    "matched_skills_count",
                    "missing_skills_count",
                    "verdict",
                    "recommendation",
                ]

                st.dataframe(
                    results_df[visible_columns],
                    use_container_width=True,
                )

                st.markdown("## Top 3 candidates")

                top3 = results_df.head(3)

                for _, row in top3.iterrows():
                    render_candidate_details(row)

                    with st.expander(f"Details for {row['candidate_name']}"):
                        st.write("Email:", row["email"])
                        st.write("Phone:", row["phone"])
                        st.write("LinkedIn:", row["linkedin"])
                        st.write("GitHub:", row["github"])

                        st.markdown("### Matched skills")
                        st.success(row["matched_skills"] if row["matched_skills"] else "No common skills found.")

                        st.markdown("### Missing skills")
                        st.warning(row["missing_skills"] if row["missing_skills"] else "No important skills missing.")

                        st.markdown("### Recommendation")
                        st.info(row["recommendation"])

                        st.markdown("### Interview questions")
                        for question in row["interview_questions"].split(" | "):
                            st.write("- " + question)

                        st.markdown("### Red flags")
                        for flag in row["red_flags"].split(" | "):
                            st.write("- " + flag)

                        st.markdown("### Agent steps")
                        st.write(row["agent_steps"])

                st.download_button(
                    label="Download CSV results",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="hr_dashboard_results.csv",
                    mime="text/csv",
                )


with tab2:
    st.markdown("## Dashboard")

    results_df = st.session_state.get("results_df")

    if results_df is None or results_df.empty:
        st.info("Run the analysis from the CV matching tab first.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Candidates analyzed", len(results_df))
        c2.metric("Average score", f"{results_df['final_score'].mean():.2f}%")
        c3.metric("Best score", f"{results_df['final_score'].max():.2f}%")
        c4.metric("Candidates above 70%", int((results_df["final_score"] >= 70).sum()))

        st.markdown("### Complete ranking")

        fig_rank = px.bar(
            results_df,
            x="candidate_name",
            y="final_score",
            hover_data=["file_name", "semantic_score", "skill_score", "verdict"],
            title="Final score per candidate",
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        st.markdown("### Score distribution")

        fig_hist = px.histogram(
            results_df,
            x="final_score",
            nbins=10,
            title="Final score distribution",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("### Matched skills vs missing skills")

        skill_df = results_df[[
            "candidate_name",
            "matched_skills_count",
            "missing_skills_count",
        ]].copy()

        skill_long = skill_df.melt(
            id_vars="candidate_name",
            value_vars=["matched_skills_count", "missing_skills_count"],
            var_name="skill_type",
            value_name="count",
        )

        fig_skills = px.bar(
            skill_long,
            x="candidate_name",
            y="count",
            color="skill_type",
            barmode="group",
            title="Matched and missing skills",
        )
        st.plotly_chart(fig_skills, use_container_width=True)

        st.markdown("### Semantic score vs skill score")

        fig_scatter = px.scatter(
            results_df,
            x="semantic_score",
            y="skill_score",
            size="final_score",
            color="verdict",
            hover_name="candidate_name",
            hover_data=["file_name", "final_score"],
            title="Semantic score vs skill score",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### HR verdicts")

        verdict_counts = results_df["verdict"].value_counts().reset_index()
        verdict_counts.columns = ["verdict", "count"]

        fig_pie = px.pie(
            verdict_counts,
            names="verdict",
            values="count",
            title="Verdict distribution",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### Full table for report")

        st.dataframe(results_df, use_container_width=True)


with tab3:
    st.markdown("## Model metrics")

    metrics_path = "reports/model_metrics.csv"

    if not os.path.exists(metrics_path):
        st.warning("reports/model_metrics.csv does not exist yet. Run: python -m src.evaluate")
    else:
        metrics_df = pd.read_csv(metrics_path)

        st.dataframe(metrics_df, use_container_width=True)

        metric_columns = [
            column for column in ["MAE", "MSE", "RMSE", "Accuracy", "Precision", "Recall", "F1_score"]
            if column in metrics_df.columns
        ]

        if metric_columns:
            selected_metric = st.selectbox("Choose metric for chart", metric_columns)

            fig_metric = px.bar(
                metrics_df,
                x="model",
                y=selected_metric,
                title=f"Model comparison by {selected_metric}",
            )

            st.plotly_chart(fig_metric, use_container_width=True)

        st.markdown("### Simple interpretation")
        st.write(
            "MAE, MSE and RMSE measure regression error between predicted score and target score. "
            "Accuracy, Precision, Recall and F1-score treat the task as classification: match or not match."
        )

        st.write(
            "If a model has high Accuracy but low F1-score, it may perform poorly on the positive match class. "
            "This is why the project reports several metrics, not only Accuracy."
        )


with tab4:
    st.markdown("## AI Agent explanation")

    st.write(
        "The system is implemented as an AI agent because it coordinates several tools in a fixed workflow. "
        "The agent receives a CV and a job description, plans the analysis steps, calls each tool, and returns an explainable result."
    )

    st.markdown("### Agent tools")

    tools_df = pd.DataFrame([
        {
            "Tool": "DocumentParserTool",
            "Role": "Extracts text from PDF, TXT and image files.",
        },
        {
            "Tool": "ApplicantInfoTool",
            "Role": "Extracts simple applicant information such as name, email, phone, LinkedIn and GitHub.",
        },
        {
            "Tool": "TextCleaningTool",
            "Role": "Normalizes and cleans CV and job description text.",
        },
        {
            "Tool": "SkillExtractionTool",
            "Role": "Detects technical and soft skills using a dictionary with aliases.",
        },
        {
            "Tool": "MatchingTool",
            "Role": "Computes semantic score, TF-IDF score, skill score and final score.",
        },
        {
            "Tool": "RecommendationTool",
            "Role": "Generates recommendations, interview questions, red flags and final verdict.",
        },
    ])

    st.dataframe(tools_df, use_container_width=True)

    st.markdown("### Agent workflow")

    st.code(
        """
extract_applicant_info
 -> clean_cv_text
 -> clean_job_text
 -> extract_cv_skills
 -> extract_job_skills
 -> compute_matching_scores
 -> generate_recommendation
 -> generate_interview_questions
 -> detect_red_flags
 -> return_explainable_result
""",
        language="text",
    )

    st.markdown("### Bias reduction")

    st.write(
        "The agent does not use protected or sensitive attributes such as age, gender, ethnicity or photo. "
        "The final decision remains with the human recruiter."
    )