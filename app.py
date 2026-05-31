import os
import pandas as pd
import streamlit as st
import plotly.express as px

from src.agent import HRAssistantAgent
from src.document_parser import extract_text_from_uploaded_file


os.makedirs("reports", exist_ok=True)

st.set_page_config(
    page_title="CV Parsing and HR Assistant",
    page_icon="HR",
    layout="wide"
)


st.markdown("""
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
""", unsafe_allow_html=True)


@st.cache_resource
def load_agent():
    return HRAssistantAgent()


agent = load_agent()


def render_candidate_details(row):
    st.markdown(
        f"""
        <div class="result-card">
            <h3>Locul {int(row["rank"])}: {row["candidate_name"]}</h3>
            <p><b>Fisier:</b> {row["file_name"]}</p>
            <p><b>Verdict:</b> {row["verdict"]}</p>
            <p><b>Scor final:</b> {row["final_score"]}%</p>
            <p><b>Scor semantic:</b> {row["semantic_score"]}%</p>
            <p><b>Skill score:</b> {row["skill_score"]}%</p>
        </div>
        """,
        unsafe_allow_html=True
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
            "verdict": result["verdict"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


st.markdown("<div class='main-title'>CV Parsing and HR Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI matching, PDF parsing, ranking, recommendations and HR analytics</div>",
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "Matching CV-uri",
    "Dashboard statistici",
    "Metrici model"
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
            "Incarca unul sau mai multe CV-uri",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            accept_multiple_files=True
        )

        if uploaded_cvs:
            st.info(f"CV-uri incarcate: {len(uploaded_cvs)}")

    with col2:
        job_text = st.text_area(
            "Job description",
            height=300,
            placeholder="Lipeste aici descrierea jobului..."
        )

    analyze_button = st.button("Analizeaza si genereaza ranking")

    if analyze_button:
        if not uploaded_cvs:
            st.warning("Incarca cel putin un CV.")
        elif not job_text.strip():
            st.warning("Completeaza descrierea jobului.")
        else:
            raw_results = []

            with st.spinner("Se extrag textele din CV-uri si se ruleaza agentul AI..."):
                for uploaded_file in uploaded_cvs:
                    cv_text = extract_text_from_uploaded_file(uploaded_file)

                    if not cv_text.strip():
                        st.warning(f"Nu s-a putut extrage text din fisierul: {uploaded_file.name}")
                        continue

                    result = agent.run(cv_text, job_text)

                    raw_results.append({
                        "file_name": uploaded_file.name,
                        "cv_text": cv_text,
                        "result": result
                    })

            if not raw_results:
                st.error("Nu s-a putut analiza niciun CV.")
            else:
                results_df = build_results_dataframe(raw_results)

                st.session_state["results_df"] = results_df
                st.session_state["raw_results"] = raw_results

                results_df.to_csv("reports/hr_dashboard_results.csv", index=False)

                st.success("Analiza a fost finalizata.")
                st.markdown("## Ranking candidati")

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
                    "recommendation"
                ]

                st.dataframe(
                    results_df[visible_columns],
                    use_container_width=True
                )

                st.markdown("## Top 3 candidati")

                top3 = results_df.head(3)

                for _, row in top3.iterrows():
                    render_candidate_details(row)

                    with st.expander(f"Detalii pentru {row['candidate_name']}"):
                        st.write("Email:", row["email"])
                        st.write("Phone:", row["phone"])
                        st.write("LinkedIn:", row["linkedin"])
                        st.write("GitHub:", row["github"])

                        st.markdown("### Skill-uri potrivite")
                        st.success(row["matched_skills"] if row["matched_skills"] else "Nu au fost gasite skill-uri comune.")

                        st.markdown("### Skill-uri lipsa")
                        st.warning(row["missing_skills"] if row["missing_skills"] else "Nu lipsesc skill-uri importante.")

                        st.markdown("### Recomandare")
                        st.info(row["recommendation"])

                        st.markdown("### Intrebari pentru interviu")
                        for question in row["interview_questions"].split(" | "):
                            st.write("- " + question)

                        st.markdown("### Red flags")
                        for flag in row["red_flags"].split(" | "):
                            st.write("- " + flag)

                st.download_button(
                    label="Descarca rezultatele CSV",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="hr_dashboard_results.csv",
                    mime="text/csv"
                )


with tab2:
    st.markdown("## Dashboard statistici")

    results_df = st.session_state.get("results_df")

    if results_df is None or results_df.empty:
        st.info("Ruleaza mai intai analiza din tab-ul Matching CV-uri.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Candidati analizati", len(results_df))
        c2.metric("Scor mediu", f"{results_df['final_score'].mean():.2f}%")
        c3.metric("Cel mai bun scor", f"{results_df['final_score'].max():.2f}%")
        c4.metric("Candidati peste 70%", int((results_df["final_score"] >= 70).sum()))

        st.markdown("### Ranking complet")

        fig_rank = px.bar(
            results_df,
            x="candidate_name",
            y="final_score",
            hover_data=["file_name", "semantic_score", "skill_score", "verdict"],
            title="Scor final per candidat"
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        st.markdown("### Distributia scorurilor")

        fig_hist = px.histogram(
            results_df,
            x="final_score",
            nbins=10,
            title="Distributia scorurilor finale"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("### Skill-uri potrivite vs skill-uri lipsa")

        skill_df = results_df[[
            "candidate_name",
            "matched_skills_count",
            "missing_skills_count"
        ]].copy()

        skill_long = skill_df.melt(
            id_vars="candidate_name",
            value_vars=["matched_skills_count", "missing_skills_count"],
            var_name="skill_type",
            value_name="count"
        )

        fig_skills = px.bar(
            skill_long,
            x="candidate_name",
            y="count",
            color="skill_type",
            barmode="group",
            title="Comparatie skill-uri potrivite/lipsa"
        )
        st.plotly_chart(fig_skills, use_container_width=True)

        st.markdown("### Relatia scor semantic - skill score")

        fig_scatter = px.scatter(
            results_df,
            x="semantic_score",
            y="skill_score",
            size="final_score",
            color="verdict",
            hover_name="candidate_name",
            hover_data=["file_name", "final_score"],
            title="Semantic score vs Skill score"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### Verdicturi HR")

        verdict_counts = results_df["verdict"].value_counts().reset_index()
        verdict_counts.columns = ["verdict", "count"]

        fig_pie = px.pie(
            verdict_counts,
            names="verdict",
            values="count",
            title="Distributia verdicturilor"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### Tabel complet pentru raport")

        st.dataframe(results_df, use_container_width=True)


with tab3:
    st.markdown("## Metrici model")

    metrics_path = "reports/model_metrics.csv"

    if not os.path.exists(metrics_path):
        st.warning("Nu exista inca reports/model_metrics.csv. Ruleaza: python -m src.evaluate")
    else:
        metrics_df = pd.read_csv(metrics_path)

        st.dataframe(metrics_df, use_container_width=True)

        metric_columns = [
            column for column in ["MAE", "MSE", "RMSE", "Accuracy", "Precision", "Recall", "F1_score"]
            if column in metrics_df.columns
        ]

        if metric_columns:
            selected_metric = st.selectbox("Alege metrica pentru grafic", metric_columns)

            fig_metric = px.bar(
                metrics_df,
                x="model",
                y=selected_metric,
                title=f"Comparatie modele dupa {selected_metric}"
            )

            st.plotly_chart(fig_metric, use_container_width=True)

        st.markdown("### Interpretare simpla")
        st.write(
            "MAE, MSE si RMSE masoara eroarea de regresie intre scorul prezis si scorul real. "
            "Accuracy, Precision, Recall si F1-score trateaza problema ca o clasificare: candidat potrivit sau nepotrivit."
        )