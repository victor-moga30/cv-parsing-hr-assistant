from pathlib import Path
from typing import Any, Dict, List

from src.final_matcher import FinalMatcher

from src.tools import (
    ApplicantInfoTool,
    DocumentParserTool,
    MatchingTool,
    RecommendationTool,
    SkillExtractionTool,
    TextCleaningTool,
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "fine_tuned_sbert"
)


class HRAssistantAgent:
    """
    Deterministic orchestration layer for
    CV-to-job analysis.

    It executes a fixed and auditable sequence
    of specialised tools. It does not autonomously
    make hiring decisions.
    """

    def __init__(self):
        if not MODEL_PATH.is_dir():
            raise FileNotFoundError(
                "The fine-tuned SBERT model is missing "
                f"from {MODEL_PATH}. "
                "Run the training pipeline before "
                "starting the application."
            )

        matcher = FinalMatcher(
            model_name=str(MODEL_PATH)
        )

        self.document_parser_tool = (
            DocumentParserTool()
        )

        self.applicant_info_tool = (
            ApplicantInfoTool()
        )

        self.cleaning_tool = (
            TextCleaningTool()
        )

        self.skill_tool = (
            SkillExtractionTool()
        )

        self.matching_tool = MatchingTool(
            matcher=matcher
        )

        self.recommendation_tool = (
            RecommendationTool()
        )

    @staticmethod
    def execution_steps(
        include_document_parsing: bool = False,
    ) -> List[str]:
        steps = []

        if include_document_parsing:
            steps.append(
                "extract_document_text"
            )

        steps.extend([
            "extract_applicant_info",
            "clean_cv_text",
            "clean_job_text",
            "extract_cv_skills",
            "extract_job_skills",
            "compute_matching_scores",
            "generate_recommendation",
            "generate_interview_questions",
            "detect_review_signals",
            "return_explainable_result",
        ])

        return steps

    def run_uploaded_file(
        self,
        uploaded_file,
        job_text: str,
    ) -> Dict[str, Any]:
        cv_text = (
            self.document_parser_tool
            .run(uploaded_file)
        )

        if not cv_text.strip():
            raise ValueError(
                "No readable text could be extracted "
                f"from {uploaded_file.name}."
            )

        return self.run(
            cv_text=cv_text,
            job_text=job_text,
            include_document_parsing=True,
        )

    def run(
        self,
        cv_text: str,
        job_text: str,
        include_document_parsing: bool = False,
    ) -> Dict[str, Any]:
        raw_cv_text = str(
            cv_text
        )

        raw_job_text = str(
            job_text
        )

        if not raw_cv_text.strip():
            raise ValueError(
                "The CV text is empty."
            )

        if not raw_job_text.strip():
            raise ValueError(
                "The job description is empty."
            )

        applicant_info = (
            self.applicant_info_tool
            .run(raw_cv_text)
        )

        clean_cv = (
            self.cleaning_tool
            .run(raw_cv_text)
        )

        clean_job = (
            self.cleaning_tool
            .run(raw_job_text)
        )

        cv_skills = (
            self.skill_tool
            .run(clean_cv)
        )

        job_skills = (
            self.skill_tool
            .run(clean_job)
        )

        result = (
            self.matching_tool
            .run(
                clean_cv=clean_cv,
                clean_job=clean_job,
                cv_skills=cv_skills,
                job_skills=job_skills,
            )
        )

        recommendation = (
            self.recommendation_tool
            .generate_recommendation(
                result
            )
        )

        interview_questions = (
            self.recommendation_tool
            .generate_interview_questions(
                matched_skills=result[
                    "matched_skills"
                ],
                missing_skills=result[
                    "missing_skills"
                ],
            )
        )

        review_signals = (
            self.recommendation_tool
            .detect_red_flags(
                result=result,
                cv_text=raw_cv_text,
            )
        )

        verdict = (
            self.recommendation_tool
            .verdict(
                result["final_score"]
            )
        )

        return {
            **result,
            "agent_steps": (
                self.execution_steps(
                    include_document_parsing
                )
            ),
            "applicant_info": applicant_info,
            "cv_skills": cv_skills,
            "job_skills": job_skills,
            "recommendation": recommendation,
            "interview_questions": (
                interview_questions
            ),
            "red_flags": review_signals,
            "verdict": verdict,
        }