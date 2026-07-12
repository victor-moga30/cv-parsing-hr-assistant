from html import escape
from pathlib import Path
from textwrap import dedent

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agent import HRAssistantAgent
from src.document_parser import (
    character_spacing_ratio,
    extract_text_from_uploaded_file,
)
from src.skill_extraction import extract_skills


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="IntelligentCVParsing | HR Assistant",
    layout="wide",
)


px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = [
    "#A855F7",
    "#22D3EE",
    "#34D399",
    "#FB7185",
    "#FBBF24",
    "#60A5FA",
]


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #020617;
    --panel: rgba(15, 23, 42, 0.72);
    --panel-strong: rgba(15, 23, 42, 0.92);
    --stroke: rgba(255, 255, 255, 0.15);
    --text: #F8FAFC;
    --muted: #CBD5E1;
    --purple: #A855F7;
    --cyan: #22D3EE;
    --pink: #FB7185;
    --green: #34D399;
    --gold: #FBBF24;
}

html,
body,
[class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 5%,
            rgba(168, 85, 247, 0.38),
            transparent 34rem
        ),
        radial-gradient(
            circle at 88% 9%,
            rgba(34, 211, 238, 0.30),
            transparent 32rem
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(251, 191, 36, 0.12),
            transparent 28rem
        ),
        linear-gradient(
            135deg,
            #020617 0%,
            #0F172A 48%,
            #111827 100%
        );

    color: var(--text);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;

    background-image:
        linear-gradient(
            rgba(255, 255, 255, 0.035) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(255, 255, 255, 0.035) 1px,
            transparent 1px
        );

    background-size: 46px 46px;
    opacity: 0.28;
    z-index: 0;
}

.block-container {
    position: relative;
    z-index: 1;

    max-width: 1320px;

    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    right: 2rem;
}

h1,
h2,
h3,
h4,
h5,
h6,
label,
p,
span,
div {
    color: var(--text);
}

hr {
    border-color: rgba(148, 163, 184, 0.18);
}


/* HERO */

.hero-shell {
    position: relative;
    overflow: hidden;

    max-width: 1220px;
    margin: 0 auto 28px auto;
    padding: 1px;

    border-radius: 34px;

    background:
        linear-gradient(
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

    min-height: 330px;
    padding: 38px 42px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 33px;
    border: 1px solid rgba(255, 255, 255, 0.12);

    background:
        linear-gradient(
            135deg,
            rgba(15, 23, 42, 0.94),
            rgba(2, 6, 23, 0.84)
        ),
        radial-gradient(
            circle at 18% 20%,
            rgba(168, 85, 247, 0.42),
            transparent 24rem
        ),
        radial-gradient(
            circle at 85% 25%,
            rgba(34, 211, 238, 0.28),
            transparent 22rem
        );
}

.hero-card::before {
    content: "";
    position: absolute;

    width: 500px;
    height: 500px;

    top: -285px;
    right: -215px;

    background:
        conic-gradient(
            from 180deg,
            rgba(168, 85, 247, 0),
            rgba(34, 211, 238, 0.40),
            rgba(251, 191, 36, 0.30),
            rgba(251, 113, 133, 0.26),
            rgba(168, 85, 247, 0)
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
        linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.09),
            transparent
        );

    transform: translateX(-80%);
    animation: heroShine 7s ease-in-out infinite;
}

@keyframes rotateGlow {
    from {
        transform: rotate(0deg);
    }

    to {
        transform: rotate(360deg);
    }
}

@keyframes heroShine {
    0%,
    55% {
        transform: translateX(-80%);
    }

    88%,
    100% {
        transform: translateX(80%);
    }
}

.hero-content {
    position: relative;
    z-index: 2;

    width: 100%;
    text-align: center;
}

.super-badge {
    width: fit-content;

    margin: 0 auto 28px auto;
    padding: 8px 14px;

    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.16);

    background: rgba(255, 255, 255, 0.08);
    color: #E0F2FE;

    font-size: 12px;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.main-title {
    margin: 0;

    text-align: center;

    font-size: clamp(44px, 4.2vw, 66px);
    line-height: 1;
    font-weight: 900;
    letter-spacing: -3px;

    background:
        linear-gradient(
            90deg,
            #C4B5FD 0%,
            #93C5FD 30%,
            #67E8F9 58%,
            #A7F3D0 100%
        );

    background-clip: text;
    -webkit-background-clip: text;

    color: transparent;
    -webkit-text-fill-color: transparent;

    filter:
        drop-shadow(
            0 12px 30px rgba(0, 0, 0, 0.28)
        );
}

.subtitle {
    max-width: 980px;

    margin: 50px auto 0 auto;

    text-align: center;
    color: #D8E3F2;

    font-size: 17px;
    line-height: 1.55;
}
.subtitle {
    max-width: 900px;
    margin: 18px auto 0 auto;

    text-align: center;
    color: #D8E3F2;

    font-size: 18px;
    line-height: 1.65;
}


/* SECTION HEADER */

.lux-card {
    margin-bottom: 18px;
    padding: 22px;

    border-radius: 26px;
    border: 1px solid rgba(255, 255, 255, 0.14);

    background:
        linear-gradient(
            145deg,
            rgba(255, 255, 255, 0.08),
            rgba(255, 255, 255, 0.035)
        ),
        rgba(15, 23, 42, 0.72);

    box-shadow:
        0 24px 70px rgba(0, 0, 0, 0.30),
        inset 0 1px 0 rgba(255, 255, 255, 0.11);

    backdrop-filter: blur(18px);
}

.section-title {
    margin-bottom: 6px;

    font-size: 30px;
    font-weight: 900;
    letter-spacing: -1px;

    background:
        linear-gradient(
            90deg,
            #FFFFFF,
            #A5F3FC,
            #DDD6FE
        );

    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
    -webkit-text-fill-color: transparent;
}

.section-subtitle {
    color: #CBD5E1;
    font-size: 15px;
}


/* TABS */

[data-testid="stTabs"] {
    margin-top: 8px;
}

[data-testid="stTabs"] button {
    padding: 12px 20px !important;

    border-radius: 999px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;

    background: rgba(15, 23, 42, 0.38) !important;
    color: #E2E8F0 !important;

    font-weight: 900 !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    background:
        linear-gradient(
            90deg,
            rgba(168, 85, 247, 0.95),
            rgba(34, 211, 238, 0.85)
        ) !important;

    color: white !important;

    box-shadow:
        0 16px 38px rgba(34, 211, 238, 0.22),
        inset 0 1px 0 rgba(255, 255, 255, 0.24);
}


/* STREAMLIT CARDS */

[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stMetric"],
.stDataFrame,
[data-testid="stExpander"] {
    border-radius: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.13) !important;

    background:
        linear-gradient(
            145deg,
            rgba(255, 255, 255, 0.075),
            rgba(255, 255, 255, 0.035)
        ),
        rgba(15, 23, 42, 0.68) !important;

    box-shadow:
        0 22px 60px rgba(0, 0, 0, 0.26),
        inset 0 1px 0 rgba(255, 255, 255, 0.10) !important;

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
    font-weight: 900;
    letter-spacing: -1px;
}


/* BUTTONS */

.stButton > button,
.stDownloadButton > button {
    padding: 13px 30px;

    border: none;
    border-radius: 18px;

    background:
        linear-gradient(
            90deg,
            #A855F7 0%,
            #22D3EE 48%,
            #FBBF24 100%
        );

    color: #020617;

    font-size: 16px;
    font-weight: 900;

    box-shadow:
        0 18px 42px rgba(34, 211, 238, 0.25),
        0 0 30px rgba(168, 85, 247, 0.18);

    transition:
        transform 0.18s ease,
        filter 0.18s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.09);
    color: #020617;
}


/* INPUTS */

.stTextArea textarea,
.stTextInput input,
[data-baseweb="select"] {
    border: 1px solid rgba(148, 163, 184, 0.30) !important;
    border-radius: 20px !important;

    background: rgba(2, 6, 23, 0.70) !important;
    color: white !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: rgba(34, 211, 238, 0.85) !important;

    box-shadow:
        0 0 0 4px rgba(34, 211, 238, 0.14) !important;
}

[data-testid="stFileUploader"] section {
    border:
        1px dashed rgba(165, 243, 252, 0.46) !important;

    border-radius: 24px !important;

    background:
        linear-gradient(
            145deg,
            rgba(168, 85, 247, 0.12),
            rgba(34, 211, 238, 0.08)
        ),
        rgba(15, 23, 42, 0.64) !important;
}

[data-testid="stFileUploader"] small {
    color: #CBD5E1 !important;
}

.stAlert {
    border-radius: 20px;
}


/* CANDIDATE CARDS */

.podium-card {
    position: relative;
    overflow: hidden;

    margin: 16px 0;
    padding: 24px;

    border-radius: 28px;
    border: 1px solid rgba(255, 255, 255, 0.16);

    background:
        linear-gradient(
            145deg,
            rgba(168, 85, 247, 0.22),
            rgba(34, 211, 238, 0.12)
        ),
        rgba(15, 23, 42, 0.82);

    box-shadow:
        0 28px 80px rgba(0, 0, 0, 0.34),
        inset 0 1px 0 rgba(255, 255, 255, 0.11);
}

.podium-card::before {
    content: "";
    position: absolute;

    width: 180px;
    height: 180px;

    right: -60px;
    top: -70px;

    background:
        radial-gradient(
            circle,
            rgba(251, 191, 36, 0.34),
            transparent 68%
        );
}

.podium-rank {
    display: inline-flex;
    align-items: center;

    padding: 8px 13px;

    border-radius: 999px;
    border: 1px solid rgba(251, 191, 36, 0.32);

    background: rgba(251, 191, 36, 0.14);
    color: #FEF3C7;

    font-size: 13px;
    font-weight: 900;
}

.podium-name {
    margin-top: 13px;
    margin-bottom: 4px;

    font-size: 27px;
    font-weight: 900;
    letter-spacing: -0.8px;
}

.podium-meta {
    margin-bottom: 14px;
    color: #CBD5E1;
    font-size: 14px;
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
    border: 1px solid rgba(34, 211, 238, 0.26);

    background: rgba(34, 211, 238, 0.12);
    color: #CFFAFE;

    font-size: 13px;
    font-weight: 900;
}

.score-pill.gold {
    border-color: rgba(251, 191, 36, 0.34);
    background: rgba(251, 191, 36, 0.15);
    color: #FEF3C7;
}

.score-pill.green {
    border-color: rgba(52, 211, 153, 0.30);
    background: rgba(52, 211, 153, 0.14);
    color: #D1FAE5;
}

.score-pill.pink {
    border-color: rgba(251, 113, 133, 0.30);
    background: rgba(251, 113, 133, 0.14);
    color: #FFE4E6;
}

.footer-note {
    margin-top: 24px;
    padding: 17px 20px;

    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.10);

    background: rgba(2, 6, 23, 0.48);
    color: #CBD5E1;

    font-size: 14px;
}
</style>
"""


def render_html(content: str) -> None:
    """
    Render HTML without allowing Streamlit Markdown
    to interpret indented tags as source code.
    """
    cleaned_content = dedent(content).strip()

    if hasattr(st, "html"):
        st.html(cleaned_content)
    else:
        # Compatibility fallback for older Streamlit versions.
        compact_content = " ".join(
            line.strip()
            for line in cleaned_content.splitlines()
        )

        st.markdown(
            compact_content,
            unsafe_allow_html=True,
        )


render_html(CUSTOM_CSS)


@st.cache_resource
def load_agent() -> HRAssistantAgent:
    return HRAssistantAgent()


try:
    agent = load_agent()

except Exception as error:
    st.error(
        "The application could not load the fine-tuned model. "
        "Run the training pipeline before starting the application."
    )

    st.exception(error)
    st.stop()


def render_page_header() -> None:
    render_html(
        """
        <div class="hero-shell">
            <div class="hero-card">
                <div class="hero-content">
                    <div class="super-badge">
                        Premium AI Recruiting Suite
                    </div>

                    <h1 class="main-title">
                        IntelligentCVParsing
                    </h1>

                    <div class="subtitle">
                        An intelligent HR assistant for CV parsing,
                        semantic matching, explainable candidate ranking,
                        interview recommendations, review signal detection,
                        and visual analytics.
                    </div>
                </div>
            </div>
        </div>
        """
    )


def section_header(
    title: str,
    subtitle: str,
) -> None:
    render_html(
        f"""
        <div class="lux-card">
            <div class="section-title">
                {escape(title)}
            </div>

            <div class="section-subtitle">
                {escape(subtitle)}
            </div>
        </div>
        """
    )


def render_candidate_details(
    row: pd.Series,
) -> None:
    rank = int(row["rank"])

    if rank == 1:
        rank_label = "Gold rank"

    elif rank == 2:
        rank_label = "Silver rank"

    else:
        rank_label = "Bronze rank"

    candidate_reference = escape(
        str(row["candidate_reference"])
    )

    file_name = escape(
        str(row["file_name"])
    )

    verdict = escape(
        str(row["verdict"])
    )

    render_html(
        f"""
        <div class="podium-card">
            <div class="podium-rank">
                {rank_label} | Rank #{rank}
            </div>

            <div class="podium-name">
                {candidate_reference}
            </div>

            <div class="podium-meta">
                <b>Name:</b> Unknown
                &nbsp; | &nbsp;
                <b>File:</b> {file_name}
                &nbsp; | &nbsp;
                <b>Verdict:</b> {verdict}
            </div>

            <div class="score-row">
                <span class="score-pill gold">
                    Final score:
                    {float(row["final_score"]):.2f} / 100
                </span>

                <span class="score-pill">
                    Semantic:
                    {float(row["semantic_score"]):.2f} / 100
                </span>

                <span class="score-pill green">
                    Skill match:
                    {float(row["skill_score"]):.2f} / 100
                </span>

                <span class="score-pill pink">
                    Missing skills:
                    {int(row["missing_skills_count"])}
                </span>
            </div>
        </div>
        """
    )


def build_results_dataframe(
    results: list,
) -> pd.DataFrame:
    rows = []

    for index, item in enumerate(
        results,
        start=1,
    ):
        result = item["result"]
        applicant_info = result.get(
            "applicant_info",
            {},
        )

        rows.append({
            "candidate_index": index,
            "candidate_reference": f"Candidate {index}",

            # The extracted name is intentionally hidden.
            "candidate_name": "Unknown",

            "file_name": item["file_name"],

            "email": applicant_info.get(
                "email",
                "Unknown",
            ),

            "phone": applicant_info.get(
                "phone",
                "Unknown",
            ),

            "linkedin": applicant_info.get(
                "linkedin",
                "Unknown",
            ),

            "github": applicant_info.get(
                "github",
                "Unknown",
            ),

            "text_length": applicant_info.get(
                "text_length",
                0,
            ),

            "final_score": float(
                result["final_score"]
            ),

            "semantic_score": float(
                result["semantic_score"]
            ),

            "baseline_score": float(
                result["baseline_score"]
            ),

            "skill_score": float(
                result["skill_score"]
            ),

            "tfidf_score": float(
                result["tfidf_score"]
            ),

            "matched_skills_count": len(
                result["matched_skills"]
            ),

            "missing_skills_count": len(
                result["missing_skills"]
            ),

            "matched_skills": ", ".join(
                result["matched_skills"]
            ),

            "missing_skills": ", ".join(
                result["missing_skills"]
            ),

            "cv_skills": ", ".join(
                result["cv_skills"]
            ),

            "job_skills": ", ".join(
                result["job_skills"]
            ),

            "recommendation": result[
                "recommendation"
            ],

            "interview_questions": " | ".join(
                result["interview_questions"]
            ),

            "review_signals": " | ".join(
                result["red_flags"]
            ),

            "agent_steps": " -> ".join(
                result.get(
                    "agent_steps",
                    [],
                )
            ),

            "verdict": result["verdict"],
        })

    dataframe = pd.DataFrame(rows)

    dataframe = dataframe.sort_values(
        "final_score",
        ascending=False,
    ).reset_index(drop=True)

    dataframe["rank"] = dataframe.index + 1

    return dataframe


def build_download_dataframe(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    return results_df.drop(
        columns=[
            "candidate_name",
            "file_name",
            "email",
            "phone",
            "linkedin",
            "github",
        ],
        errors="ignore",
    )


def transparent_figure(
    figure,
):
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "#F8FAFC",
        },
    )

    return figure


render_page_header()


tab1, tab2, tab3, tab4 = st.tabs([
    "CV Matching Arena",
    "Executive Dashboard",
    "Model Performance",
    "Agent Logic",
])


if "results_df" not in st.session_state:
    st.session_state["results_df"] = None


with tab1:
    section_header(
        "Candidate Matching Arena",
        (
            "Upload candidate CVs, add the job description, "
            "and let the AI assistant build an explainable ranking."
        ),
    )

    column_cv, column_job = st.columns(
        [1, 1],
        gap="large",
    )

    with column_cv:
        st.markdown("### Upload CV files")

        uploaded_cvs = st.file_uploader(
            "Upload one or more CV files",
            type=[
                "pdf",
                "png",
                "jpg",
                "jpeg",
                "txt",
            ],
            accept_multiple_files=True,
        )

        show_extracted_text = st.checkbox(
            "Show extracted text and detected skills",
            value=False,
        )

        if uploaded_cvs:
            st.success(
                f"Uploaded CV files: {len(uploaded_cvs)}"
            )

    with column_job:
        st.markdown("### Job description")

        job_text = st.text_area(
            "Job description",
            height=310,
            placeholder="Paste the job description here...",
        )

        if job_text.strip():
            st.markdown(
                "#### Detected job skills"
            )

            detected_job_skills = extract_skills(
                job_text
            )

            if detected_job_skills:
                st.write(
                    ", ".join(
                        detected_job_skills
                    )
                )

            else:
                st.write(
                    "No dictionary skills detected."
                )

    st.markdown("---")

    analyze_button = st.button(
        "Analyze candidates and generate ranking"
    )

    if analyze_button:
        if not uploaded_cvs:
            st.warning(
                "Upload at least one CV."
            )

        elif not job_text.strip():
            st.warning(
                "Complete the job description."
            )

        else:
            raw_results = []

            with st.spinner(
                "The AI assistant is extracting, comparing, "
                "and ranking candidates..."
            ):
                for uploaded_file in uploaded_cvs:
                    try:
                        cv_text = (
                            extract_text_from_uploaded_file(
                                uploaded_file
                            )
                        )

                        if not cv_text.strip():
                            st.warning(
                                "Could not extract text from file: "
                                f"{uploaded_file.name}"
                            )

                            continue

                        if (
                            character_spacing_ratio(
                                cv_text
                            )
                            > 0.35
                        ):
                            st.warning(
                                "The text extracted from "
                                f"{uploaded_file.name} "
                                "still contains unusually separated characters."
                            )

                        result = agent.run(
                            cv_text=cv_text,
                            job_text=job_text,
                            include_document_parsing=True,
                        )

                        if show_extracted_text:
                            with st.expander(
                                f"Extracted text from {uploaded_file.name}",
                                expanded=False,
                            ):
                                st.text(
                                    cv_text[:5000]
                                )

                                st.markdown(
                                    "### Detected CV skills"
                                )

                                if result["cv_skills"]:
                                    st.write(
                                        ", ".join(
                                            result[
                                                "cv_skills"
                                            ]
                                        )
                                    )

                                else:
                                    st.write(
                                        "No dictionary skills detected."
                                    )

                        raw_results.append({
                            "file_name": uploaded_file.name,
                            "result": result,
                        })

                    except Exception as error:
                        st.error(
                            "Could not analyse "
                            f"{uploaded_file.name}: {error}"
                        )

            if not raw_results:
                st.error(
                    "No CV could be analysed."
                )

            else:
                results_df = build_results_dataframe(
                    raw_results
                )

                st.session_state[
                    "results_df"
                ] = results_df

                st.success(
                    "Analysis completed. "
                    "Ranking generated successfully."
                )

                st.markdown(
                    "## Final ranking"
                )

                visible_columns = [
                    "rank",
                    "candidate_reference",
                    "candidate_name",
                    "final_score",
                    "semantic_score",
                    "skill_score",
                    "matched_skills_count",
                    "missing_skills_count",
                    "verdict",
                ]

                ranking_dataframe = results_df[
                    visible_columns
                ].rename(
                    columns={
                        "candidate_reference": (
                            "candidate"
                        ),
                    }
                )

                st.dataframe(
                    ranking_dataframe,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown(
                    "## Top 3 candidates"
                )

                for _, row in (
                    results_df
                    .head(3)
                    .iterrows()
                ):
                    render_candidate_details(
                        row
                    )

                    with st.expander(
                        "Open full analysis for "
                        f"{row['candidate_reference']}"
                    ):
                        st.write(
                            "**Candidate name:** Unknown"
                        )

                        st.write(
                            "**Email:**",
                            row["email"],
                        )

                        st.write(
                            "**Phone:**",
                            row["phone"],
                        )

                        st.write(
                            "**LinkedIn:**",
                            row["linkedin"],
                        )

                        st.write(
                            "**GitHub:**",
                            row["github"],
                        )

                        st.markdown(
                            "### Matched skills"
                        )

                        st.success(
                            row["matched_skills"]
                            if row["matched_skills"]
                            else (
                                "No common skills were found."
                            )
                        )

                        st.markdown(
                            "### Missing skills"
                        )

                        st.warning(
                            row["missing_skills"]
                            if row["missing_skills"]
                            else (
                                "No important skills are missing."
                            )
                        )

                        st.markdown(
                            "### Recommendation"
                        )

                        st.info(
                            row["recommendation"]
                        )

                        st.markdown(
                            "### Interview questions"
                        )

                        for question in (
                            row["interview_questions"]
                            .split(" | ")
                        ):
                            st.write(
                                f"- {question}"
                            )

                        st.markdown(
                            "### Review signals"
                        )

                        for signal in (
                            row["review_signals"]
                            .split(" | ")
                        ):
                            st.write(
                                f"- {signal}"
                            )

                        st.markdown(
                            "### Agent steps"
                        )

                        st.write(
                            row["agent_steps"]
                        )

                download_dataframe = (
                    build_download_dataframe(
                        results_df
                    )
                )

                st.download_button(
                    label="Download CSV results",
                    data=(
                        download_dataframe
                        .to_csv(index=False)
                        .encode("utf-8")
                    ),
                    file_name="hr_dashboard_results.csv",
                    mime="text/csv",
                )


with tab2:
    section_header(
        "Executive HR Dashboard",
        (
            "Quick visual insights for candidate scores, "
            "distributions, verdicts, and skill gaps."
        ),
    )

    results_df = st.session_state.get(
        "results_df"
    )

    if (
        results_df is None
        or results_df.empty
    ):
        st.info(
            "Run the analysis from the "
            "CV Matching Arena tab first."
        )

    else:
        metric_1, metric_2, metric_3, metric_4 = (
            st.columns(4)
        )

        metric_1.metric(
            "Candidates analyzed",
            len(results_df),
        )

        metric_2.metric(
            "Average score",
            (
                f"{results_df['final_score'].mean():.2f}"
                " / 100"
            ),
        )

        metric_3.metric(
            "Best score",
            (
                f"{results_df['final_score'].max():.2f}"
                " / 100"
            ),
        )

        metric_4.metric(
            "Candidates above 70",
            int(
                (
                    results_df["final_score"]
                    >= 70
                ).sum()
            ),
        )

        st.markdown(
            "### Ranking by final score"
        )

        ranking_figure = px.bar(
            results_df,
            x="candidate_reference",
            y="final_score",
            hover_data=[
                "semantic_score",
                "skill_score",
                "verdict",
            ],
            title="Final score per candidate",
            labels={
                "candidate_reference": "Candidate",
                "final_score": "Final score",
            },
        )

        st.plotly_chart(
            transparent_figure(
                ranking_figure
            ),
            use_container_width=True,
        )

        st.markdown(
            "### Final score distribution"
        )

        histogram_figure = px.histogram(
            results_df,
            x="final_score",
            nbins=10,
            title="Final score distribution",
        )

        st.plotly_chart(
            transparent_figure(
                histogram_figure
            ),
            use_container_width=True,
        )

        st.markdown(
            "### Matched skills vs missing skills"
        )

        skills_dataframe = results_df[[
            "candidate_reference",
            "matched_skills_count",
            "missing_skills_count",
        ]].copy()

        long_skills_dataframe = (
            skills_dataframe.melt(
                id_vars="candidate_reference",
                value_vars=[
                    "matched_skills_count",
                    "missing_skills_count",
                ],
                var_name="skill_type",
                value_name="count",
            )
        )

        skills_figure = px.bar(
            long_skills_dataframe,
            x="candidate_reference",
            y="count",
            color="skill_type",
            barmode="group",
            title="Matched and missing skills",
            labels={
                "candidate_reference": "Candidate",
                "count": "Skills",
            },
        )

        st.plotly_chart(
            transparent_figure(
                skills_figure
            ),
            use_container_width=True,
        )

        st.markdown(
            "### Semantic score vs skill score"
        )

        scatter_figure = px.scatter(
            results_df,
            x="semantic_score",
            y="skill_score",
            size="final_score",
            color="verdict",
            hover_name="candidate_reference",
            title="Semantic score vs skill score",
        )

        st.plotly_chart(
            transparent_figure(
                scatter_figure
            ),
            use_container_width=True,
        )

        st.markdown(
            "### Verdict distribution"
        )

        verdict_counts = (
            results_df["verdict"]
            .value_counts()
            .reset_index()
        )

        verdict_counts.columns = [
            "verdict",
            "count",
        ]

        verdict_figure = px.pie(
            verdict_counts,
            names="verdict",
            values="count",
            title="Verdict distribution",
            hole=0.42,
        )

        st.plotly_chart(
            transparent_figure(
                verdict_figure
            ),
            use_container_width=True,
        )

        st.markdown(
            "### Full table for report"
        )

        dashboard_columns = [
            "rank",
            "candidate_reference",
            "candidate_name",
            "final_score",
            "semantic_score",
            "baseline_score",
            "tfidf_score",
            "skill_score",
            "matched_skills",
            "missing_skills",
            "verdict",
        ]

        st.dataframe(
            results_df[
                dashboard_columns
            ],
            use_container_width=True,
            hide_index=True,
        )


with tab3:
    section_header(
        "Model Performance Lab",
        (
            "Compare the models and metrics used "
            "to evaluate the matching system."
        ),
    )

    metrics_path = (
        REPORTS_DIR
        / "model_metrics.csv"
    )

    if not metrics_path.exists():
        st.warning(
            "reports/model_metrics.csv does not exist yet. "
            "Run: python -m src.evaluate"
        )

    else:
        metrics_df = pd.read_csv(
            metrics_path
        )

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
        )

        metric_columns = [
            column
            for column in [
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
            selected_metric = st.selectbox(
                "Choose metric for chart",
                metric_columns,
            )

            metric_figure = px.bar(
                metrics_df,
                x="model",
                y=selected_metric,
                title=(
                    "Model comparison by "
                    f"{selected_metric}"
                ),
            )

            st.plotly_chart(
                transparent_figure(
                    metric_figure
                ),
                use_container_width=True,
            )

        st.markdown(
            "### Simple interpretation"
        )

        st.write(
            "MAE, MSE and RMSE measure regression "
            "error between the predicted score and "
            "the target score. Accuracy, Precision, "
            "Recall and F1-score treat the task as "
            "classification: match or not match."
        )

        st.write(
            "If a model has high Accuracy but low "
            "F1-score, it may perform poorly on the "
            "positive match class. This is why the "
            "project reports several metrics rather "
            "than only Accuracy."
        )


with tab4:
    section_header(
        "Agent Logic",
        (
            "A clear explanation of how the orchestration "
            "layer uses specialised tools to produce the result."
        ),
    )

    st.write(
        "The system uses a deterministic orchestration "
        "layer that coordinates several specialised "
        "tools in a fixed and auditable workflow. "
        "It receives a CV and a job description, "
        "executes each step, and returns an explainable result."
    )

    st.markdown(
        "### Agent tools"
    )

    tools_dataframe = pd.DataFrame([
        {
            "Tool": "DocumentParserTool",
            "Role": (
                "Extracts text from PDF, TXT and image files."
            ),
        },
        {
            "Tool": "ApplicantInfoTool",
            "Role": (
                "Extracts contact information for optional review."
            ),
        },
        {
            "Tool": "TextCleaningTool",
            "Role": (
                "Normalizes and cleans the CV and job text."
            ),
        },
        {
            "Tool": "SkillExtractionTool",
            "Role": (
                "Detects skills using controlled aliases."
            ),
        },
        {
            "Tool": "MatchingTool",
            "Role": (
                "Computes semantic, TF-IDF, skill and final scores."
            ),
        },
        {
            "Tool": "RecommendationTool",
            "Role": (
                "Produces recommendations, questions and review signals."
            ),
        },
    ])

    st.dataframe(
        tools_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Agent workflow"
    )

    st.code(
        dedent(
            """
            extract_document_text
             -> extract_applicant_info
             -> clean_cv_text
             -> clean_job_text
             -> extract_cv_skills
             -> extract_job_skills
             -> compute_matching_scores
             -> generate_recommendation
             -> generate_interview_questions
             -> detect_review_signals
             -> return_explainable_result
            """
        ).strip(),
        language="text",
    )

    st.markdown(
        "### Bias reduction"
    )

    st.write(
        "Candidate names are hidden from ranking and "
        "are not used as scoring features. Protected "
        "attributes are not assigned as dedicated "
        "features, and contact fields are removed from "
        "the cleaned matching representation. However, "
        "CV language may still contain indirect proxies, "
        "so the prototype is not presented as bias-free. "
        "The final decision remains with the human recruiter."
    )

    render_html(
        """
        <div class="footer-note">
            <b>Note:</b>
            This interface is designed as a decision-support
            assistant, not as an automatic hiring system.
            Scores are compatibility indicators out of 100,
            not probabilities of hiring success. The final
            decision must remain human and explainable.
        </div>
        """
    )