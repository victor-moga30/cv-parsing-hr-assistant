from pathlib import Path
from typing import Any, Dict, List

from src.final_matcher import FinalMatcher
from src.tools import (
    ApplicantInfoTool,
    TextCleaningTool,
    SkillExtractionTool,
    MatchingTool,
    RecommendationTool,
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fine_tuned_sbert"


class HRAssistantAgent:
    """
    AI Agent for CV parsing and HR matching.

    The agent does not make a hiring decision.
    It coordinates several internal tools and produces an explainable support result.
    """

    def __init__(self):
        if MODEL_PATH.exists():
            matcher = FinalMatcher(model_name=str(MODEL_PATH))
        else:
            matcher = FinalMatcher()

        self.applicant_info_tool = ApplicantInfoTool()
        self.cleaning_tool = TextCleaningTool()
        self.skill_tool = SkillExtractionTool()
        self.matching_tool = MatchingTool(matcher=matcher)
        self.recommendation_tool = RecommendationTool()

    def plan_steps(self) -> List[str]:
        return [
            "extract_applicant_info",
            "clean_cv_text",
            "clean_job_text",
            "extract_cv_skills",
            "extract_job_skills",
            "compute_matching_scores",
            "generate_recommendation",
            "generate_interview_questions",
            "detect_red_flags",
            "return_explainable_result",
        ]

    def run(self, cv_text: str, job_text: str) -> Dict[str, Any]:
        raw_cv_text = str(cv_text)
        raw_job_text = str(job_text)

        applicant_info = self.applicant_info_tool.run(raw_cv_text)

        clean_cv = self.cleaning_tool.run(raw_cv_text)
        clean_job = self.cleaning_tool.run(raw_job_text)

        cv_skills = self.skill_tool.run(raw_cv_text + " " + clean_cv)
        job_skills = self.skill_tool.run(raw_job_text + " " + clean_job)

        result = self.matching_tool.run(
            clean_cv=clean_cv,
            clean_job=clean_job,
            cv_skills=cv_skills,
            job_skills=job_skills,
        )

        recommendation = self.recommendation_tool.generate_recommendation(result)

        interview_questions = self.recommendation_tool.generate_interview_questions(
            matched_skills=result["matched_skills"],
            missing_skills=result["missing_skills"],
        )

        red_flags = self.recommendation_tool.detect_red_flags(
            result=result,
            cv_text=raw_cv_text,
        )

        verdict = self.recommendation_tool.verdict(result["final_score"])

        return {
            **result,
            "agent_steps": self.plan_steps(),
            "applicant_info": applicant_info,
            "cv_skills": cv_skills,
            "job_skills": job_skills,
            "recommendation": recommendation,
            "interview_questions": interview_questions,
            "red_flags": red_flags,
            "verdict": verdict,
        }