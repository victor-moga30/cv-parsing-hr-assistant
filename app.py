import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agent import HRAssistantAgent
from src.document_parser import extract_text_from_uploaded_file
from src.skill_extraction import extract_skills


os.makedirs("reports", exist_ok=True)

px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = [
    "#A855F7",
    "#22D3EE",
    "#34D399",
    "#FB7185",
    "#FBBF24",
    "#60A5FA",
]

st.set_page_config(
    page_title="IntelligentCVParsing | HR Assistant",
    layout="wide",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #020617;
    --panel: rgba(15, 23, 42, 0.72);
    --panel-strong: rgba(15, 23, 42, 0.92);
    --glass: rgba(255, 255, 255, 0.065);
    --stroke: rgba(255, 255, 255, 0.15);
    --stroke-2: rgba(148, 163, 184, 0.22);
    --text: #F8FAFC;
    --muted: #CBD5E1;
    --muted-2: #94A3B8;
    --purple: #A855F7;
    --cyan: #22D3EE;
    --pink: #FB7185;
    --green: #34D399;
    --gold: #FBBF24;
}

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(168, 85, 247, 0.38), transparent 34rem),
        radial-gradient(circle at 88% 9%, rgba(34, 211, 238, 0.30), transparent 32rem),
        radial-gradient(circle at 50% 100%, rgba(251, 191, 36, 0.12), transparent 28rem),
        linear-gradient(135deg, #020617 0%, #0F172A 48%, #111827 100%);
    color: var(--text);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
    background-size: 46px 46px;
    mask-image: radial-gradient(circle at center, black 0%, transparent 72%);
    opacity: 0.35;
    z-index: 0;
}

.block-container {
    position: relative;
    z-index: 1;
    padding-top: 2.1rem;
    padding-bottom: 3rem;
    max-width: 1320px;
}

[data-testid="stHeader"] {
    background: rgba(2, 6, 23, 0);
}

[data-testid="stToolbar"] {
    right: 2rem;
}

h1, h2, h3, h4, h5, h6, label, p, span, div {
    color: var(--text);
}

hr {
    border-color: rgba(148, 163, 184, 0.18);
}

/* HERO */

.hero-shell {
    position: relative;
    overflow: hidden;
    padding: 1px;
    border-radius: 34px;
    margin-bottom: 28px;
    background: linear-gradient(
        135deg,
        rgba(168, 85, 247, 0.85),
        rgba(34, 211, 238, 0.72),
        rgba(251, 191, 36, 0.50),
        rgba(251, 113, 133, 0.68)
    );
    box-shadow:
        0 28px 90px rgba(0, 0, 0, 0.48),
        0 0 80px rgba(168, 85, 247, 0.16);
}

.hero-card {
    position: relative;
    overflow: hidden;
    border-radius: 33px;
    padding: 42px 38px 34px 38px;
    background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(2, 6, 23, 0.84)),
        radial-gradient(circle at 18% 20%, rgba(168, 85, 247, 0.42), transparent 24rem),
        radial-gradient(circle at 85% 25%, rgba(34, 211, 238, 0.28), transparent 22rem);
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.hero-card::before {
    content: "";
    position: absolute;
    width: 560px;
    height: 560px;
    top: -310px;
    right: -250px;
    background: conic-gradient(
        from 180deg,
        rgba(168, 85, 247, 0.0),
        rgba(34, 211, 238, 0.40),
        rgba(251, 191, 36, 0.30),
        rgba(251, 113, 133, 0.26),
        rgba(168, 85, 247, 0.0)
    );
    filter: blur(8px);
    opacity: 0.9;
    animation: rotateGlow 15s linear infinite;
}

.hero-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, transparent, rgba(255,255,255,0.09), transparent);
    transform: translateX(-80%);
    animation: heroShine 7s ease-in-out infinite;
}

@keyframes rotateGlow {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes heroShine {
    0%, 55% { transform: translateX(-80%); }
    88%, 100% { transform: translateX(80%); }
}

.hero-content {
    position: relative;
    z-index: 2;
}

.super-badge {
    width: fit-content;
    margin: 0 auto 16px auto;
    padding: 9px 15px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.16);
    color: #E0F2FE;
    font-size: 13px;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.12);
}

.main-title {
    text-align: center;
    font-size: clamp(46px, 7vw, 88px);
    line-height: 0.93;
    font-weight: 950;
    letter-spacing: -3px;
    margin: 0;
    background: linear-gradient(90deg, #FFFFFF 0%, #C4B5FD 28%, #67E8F9 55%, #FDE68A 78%, #FDA4AF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 16px 36px rgba(0,0,0,0.38));
}

.subtitle {
    text-align: center;
    font-size: 18px;
    line-height: 1.65;
    color: #D8E3F2;
    max-width: 900px;
    margin: 18px auto 0 auto;
}

/* SECTION CARDS */

.lux-card {
    padding: 22px;
    border-radius: 26px;
    background:
        linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.035)),
        rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow:
        0 24px 70px rgba(0,0,0,0.30),
        inset 0 1px 0 rgba(255,255,255,0.11);
    backdrop-filter: blur(18px);
    margin-bottom: 18px;
}

.section-title {
    font-size: 30px;
    font-weight: 950;
    letter-spacing: -1px;
    margin-bottom: 6px;
    background: linear-gradient(90deg, #FFFFFF, #A5F3FC, #DDD6FE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section-subtitle {
    color: #CBD5E1;
    font-size: 15px;
    margin-bottom: 18px;
}

/* STREAMLIT BASE */

[data-testid="stTabs"] {
    margin-top: 8px;
}

[data-testid="stTabs"] button {
    border-radius: 999px !important;
    padding: 12px 20px !important;
    font-weight: 900 !important;
    color: #E2E8F0 !important;
    background: rgba(15, 23, 42, 0.38) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    background:
        linear-gradient(90deg, rgba(168, 85, 247, 0.95), rgba(34, 211, 238, 0.85)) !important;
    color: white !important;
    box-shadow:
        0 16px 38px rgba(34, 211, 238, 0.22),
        inset 0 1px 0 rgba(255,255,255,0.24);
}

[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stMetric"],
.stDataFrame,
[data-testid="stExpander"] {
    border-radius: 24px !important;
    background:
        linear-gradient(145deg, rgba(255,255,255,0.075), rgba(255,255,255,0.035)),
        rgba(15, 23, 42, 0.68) !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    box-shadow:
        0 22px 60px rgba(0, 0, 0, 0.26),
        inset 0 1px 0 rgba(255,255,255,0.10) !important;
    backdrop-filter: blur(16px);
}

[data-testid="stMetric"] {
    padding: 20px 19px 15px 19px;
}

[data-testid="stMetricLabel"] p {
    color: #CBD5E1 !important;
    font-weight: 800;
}

[data-testid="stMetricValue"] div {
    color: #A5F3FC !important;
    font-weight: 950;
    letter-spacing: -1px;
}

.stButton > button, .stDownloadButton > button {
    position: relative;
    background: linear-gradient(90deg, #A855F7 0%, #22D3EE 48%, #FBBF24 100%);
    color: #020617;
    border: none;
    border-radius: 18px;
    padding: 13px 30px;
    font-weight: 950;
    font-size: 16px;
    box-shadow:
        0 18px 42px rgba(34, 211, 238, 0.25),
        0 0 30px rgba(168, 85, 247, 0.18);
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    filter: brightness(1.09);
    box-shadow:
        0 24px 60px rgba(168, 85, 247, 0.30),
        0 0 40px rgba(34, 211, 238, 0.22);
    color: #020617;
}

.stTextArea textarea, .stTextInput input, [data-baseweb="select"] {
    background: rgba(2, 6, 23, 0.70) !important;
    color: white !important;
    border: 1px solid rgba(148, 163, 184, 0.30) !important;
    border-radius: 20px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(34, 211, 238, 0.85) !important;
    box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.14) !important;
}

[data-testid="stFileUploader"] section {
    background:
        linear-gradient(145deg, rgba(168, 85, 247, 0.12), rgba(34, 211, 238, 0.08)),
        rgba(15, 23, 42, 0.64) !important;
    border: 1px dashed rgba(165, 243, 252, 0.46) !important;
    border-radius: 24px !important;
}

[data-testid="stFileUploader"] small {
    color: #CBD5E1 !important;
}

.stAlert {
    border-radius: 20px;
}

/* CUSTOM CANDIDATE CARDS */

.podium-card {
    position: relative;
    overflow: hidden;
    padding: 24px;
    border-radius: 28px;
    margin: 16px 0;
    background:
        linear-gradient(145deg, rgba(168, 85, 247, 0.22), rgba(34, 211, 238, 0.12)),
        rgba(15, 23, 42, 0.82);
    border: 1px solid rgba(255,255,255,0.16);
    box-shadow:
        0 28px 80px rgba(0,0,0,0.34),
        inset 0 1px 0 rgba(255,255,255,0.11);
}

.podium-card::before {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    right: -60px;
    top: -70px;
    background: radial-gradient(circle, rgba(251, 191, 36, 0.34), transparent 68%);
}

.podium-rank {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(251, 191, 36, 0.14);
    border: 1px solid rgba(251, 191, 36, 0.32);
    color: #FEF3C7;
    font-size: 13px;
    font-weight: 950;
}

.podium-name {
    margin-top: 13px;
    margin-bottom: 4px;
    font-size: 27px;
    font-weight: 950;
    letter-spacing: -0.8px;
}

.podium-meta {
    color: #CBD5E1;
    font-size: 14px;
    margin-bottom: 14px;
}

.score-row {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 14px;
}

.score-pill {
    display: inline-block;
    padding: 9px 13px;
    border-radius: 999px;
    background: rgba(34, 211, 238, 0.12);
    border: 1px solid rgba(34, 211, 238, 0.26);
    color: #CFFAFE;
    font-weight: 900;
    font-size: 13px;
}

.score-pill.gold {
    background: rgba(251, 191, 36, 0.15);
    border-color: rgba(251, 191, 36, 0.34);
    color: #FEF3C7;
}

.score-pill.green {
    background: rgba(52, 211, 153, 0.14);
    border-color: rgba(52, 211, 153, 0.30);
    color: #D1FAE5;
}

.score-pill.pink {
    background: rgba(251, 113, 133, 0.14);
    border-color: rgba(251, 113, 133, 0.30);
    color: #FFE4E6;
}

.small-muted {
    color: var(--muted);
    font-size: 14px;
}

.footer-note {
    margin-top: 24px;
    padding: 17px 20px;
    border-radius: 22px;
    background: rgba(2, 6, 23, 0.48);
    border: 1px solid rgba(255,255,255,0.10);
    color: #CBD5E1;
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


def render_page_header():
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-card">
                <div class="hero-content">
                    <div class="super-badge">Premium AI Recruiting Suite</div>
                    <h1 class="main-title">IntelligentCVParsing</h1>
                    <div class="subtitle">
                        An intelligent HR assistant for CV parsing, semantic matching,
                        explainable candidate ranking, interview recommendations,
                        red flag detection, and visual analytics.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle):
    st.markdown(
        f"""
        <div class="lux-card">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_candidate_details(row):
    rank_label = "Gold rank" if int(row["rank"]) == 1 else "Silver rank" if int(row["rank"]) == 2 else "Bronze rank"

    st.markdown(
        f"""
        <div class="podium-card">
            <div class="podium-rank">{rank_label} | Rank #{int(row["rank"])}</div>
            <div class="podium-name">{row["candidate_name"]}</div>
            <div class="podium-meta">
                <b>File:</b> {row["file_name"]} &nbsp; | &nbsp;
                <b>Verdict:</b> {row["verdict"]}
            </div>
            <div class="score-row">
                <span class="score-pill gold">Final score: {row["final_score"]}%</span>
                <span class="score-pill">Semantic: {row["semantic_score"]}%</span>
                <span class="score-pill green">Skill match: {row["skill_score"]}%</span>
                <span class="score-pill pink">Missing skills: {row["missing_skills_count"]}</span>
            </div>
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


render_page_header()

tab1, tab2, tab3, tab4 = st.tabs([
    "CV Matching Arena",
    "Executive Dashboard",
    "Model Performance",
    "Agent Logic",
])

if "results_df" not in st.session_state:
    st.session_state["results_df"] = None

if "raw_results" not in st.session_state:
    st.session_state["raw_results"] = None


with tab1:
    section_header(
        "Candidate Matching Arena",
        "Upload candidate CVs, add the job description, and let the AI agent build an explainable ranking."
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Upload CV files")

        uploaded_cvs = st.file_uploader(
            "Upload one or more CV files",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            accept_multiple_files=True,
        )

        show_extracted_text = st.checkbox(
            "Show extracted text and detected skills",
            value=False,
        )

        if uploaded_cvs:
            st.success(f"Uploaded CV files: {len(uploaded_cvs)}")

    with col2:
        st.markdown("### Job description")

        job_text = st.text_area(
            "Job description",
            height=310,
            placeholder="Paste the job description here...",
        )

        if job_text.strip():
            st.markdown("#### Detected job skills")
            st.write(extract_skills(job_text))

    st.markdown("---")

    analyze_button = st.button("Analyze candidates and generate ranking")

    if analyze_button:
        if not uploaded_cvs:
            st.warning("Upload at least one CV.")
        elif not job_text.strip():
            st.warning("Complete the job description.")
        else:
            raw_results = []

            with st.spinner("The AI agent is extracting, comparing, and ranking candidates..."):
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

                st.success("Analysis completed. Ranking generated successfully.")

                st.markdown("## Final ranking")

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

                    with st.expander(f"Open full analysis for {row['candidate_name']}"):
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
    section_header(
        "Executive HR Dashboard",
        "Quick visual insights for candidate scores, distributions, verdicts, and skill gaps."
    )

    results_df = st.session_state.get("results_df")

    if results_df is None or results_df.empty:
        st.info("Run the analysis from the CV Matching Arena tab first.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Candidates analyzed", len(results_df))
        c2.metric("Average score", f"{results_df['final_score'].mean():.2f}%")
        c3.metric("Best score", f"{results_df['final_score'].max():.2f}%")
        c4.metric("Candidates above 70%", int((results_df["final_score"] >= 70).sum()))

        st.markdown("### Ranking by final score")

        fig_rank = px.bar(
            results_df,
            x="candidate_name",
            y="final_score",
            hover_data=["file_name", "semantic_score", "skill_score", "verdict"],
            title="Final score per candidate",
        )
        fig_rank.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        st.markdown("### Final score distribution")

        fig_hist = px.histogram(
            results_df,
            x="final_score",
            nbins=10,
            title="Final score distribution",
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
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
        fig_skills.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
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
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### Verdict distribution")

        verdict_counts = results_df["verdict"].value_counts().reset_index()
        verdict_counts.columns = ["verdict", "count"]

        fig_pie = px.pie(
            verdict_counts,
            names="verdict",
            values="count",
            title="Verdict distribution",
            hole=0.42,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### Full table for report")
        st.dataframe(results_df, use_container_width=True)


with tab3:
    section_header(
        "Model Performance Lab",
        "Compare the models and metrics used to evaluate the matching system."
    )

    metrics_path = "reports/model_metrics.csv"

    if not os.path.exists(metrics_path):
        st.warning("reports/model_metrics.csv does not exist yet. Run: python -m src.evaluate")
    else:
        metrics_df = pd.read_csv(metrics_path)

        st.dataframe(metrics_df, use_container_width=True)

        metric_columns = [
            column for column in [
                "MAE",
                "MSE",
                "RMSE",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_score",
            ]
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
            fig_metric.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
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
    section_header(
        "Agent Logic",
        "A clear explanation of how the AI agent uses tools to produce the final result."
    )

    st.write(
        "The system is implemented as an AI agent because it coordinates several tools in a fixed workflow. "
        "The agent receives a CV and a job description, plans the analysis steps, calls each tool, "
        "and returns an explainable result."
    )

    st.markdown("### Agent tools")

    tools_df = pd.DataFrame([
        {
            "Tool": "DocumentParserTool",
            "Role": "Extracts text from PDF, TXT and image files.",
        },
        {
            "Tool": "ApplicantInfoTool",
            "Role": "Extracts applicant information such as name, email, phone, LinkedIn and GitHub.",
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

    st.markdown(
        """
        <div class="footer-note">
            <b>Note:</b> This interface is designed as a decision-support assistant, not as an automatic hiring system.
            It helps HR teams compare candidates faster, but the final decision must remain human and explainable.
        </div>
        """,
        unsafe_allow_html=True,
    )