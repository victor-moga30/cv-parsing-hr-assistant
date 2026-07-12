from typing import Any, Dict, List

from src.document_parser import (
    extract_applicant_info,
    extract_text_from_uploaded_file,
)

from src.final_matcher import FinalMatcher
from src.preprocessing import clean_text
from src.skill_extraction import extract_skills


STRONG_MATCH_THRESHOLD = 80.0
GOOD_MATCH_THRESHOLD = 60.0
MEDIUM_MATCH_THRESHOLD = 40.0


class DocumentParserTool:
    """
    Extract text from PDF, TXT or image files.
    """

    def run(self, uploaded_file) -> str:
        if uploaded_file is None:
            return ""

        return extract_text_from_uploaded_file(
            uploaded_file
        )


class ApplicantInfoTool:
    """
    Extract contact information for display.

    These values are separated from the cleaned
    representation used by the scoring pipeline.
    """

    def run(
        self,
        cv_text: str,
    ) -> Dict[str, Any]:
        return extract_applicant_info(
            cv_text
        )


class TextCleaningTool:
    """
    Remove contact fields and normalise text
    before scoring.
    """

    def run(
        self,
        text: str,
    ) -> str:
        return clean_text(
            text
        )


class SkillExtractionTool:
    """
    Detect technical and soft skills using
    a controlled alias dictionary.
    """

    def run(
        self,
        text: str,
    ) -> List[str]:
        return extract_skills(
            text
        )


class MatchingTool:
    """
    Compute semantic, lexical, skill and
    final hybrid scores.
    """

    def __init__(
        self,
        matcher: FinalMatcher,
    ):
        self.matcher = matcher

    def run(
        self,
        clean_cv: str,
        clean_job: str,
        cv_skills: List[str],
        job_skills: List[str],
    ) -> Dict[str, Any]:
        return self.matcher.match(
            cv_text=clean_cv,
            job_text=clean_job,
            cv_skills=cv_skills,
            job_skills=job_skills,
        )


class RecommendationTool:
    """
    Generate deterministic next-step suggestions,
    interview questions and review signals.

    This tool does not make an autonomous hiring
    decision.
    """

    @staticmethod
    def _matched_text(
        matched_skills: List[str],
    ) -> str:
        if matched_skills:
            return ", ".join(
                matched_skills[:8]
            )

        return (
            "no clearly detected common "
            "technical skills"
        )

    @staticmethod
    def _missing_text(
        missing_skills: List[str],
    ) -> str:
        if missing_skills:
            return ", ".join(
                missing_skills[:5]
            )

        return (
            "no major missing requirements "
            "detected"
        )

    def generate_recommendation(
        self,
        result: Dict[str, Any],
    ) -> str:
        score = float(
            result["final_score"]
        )

        matched_text = self._matched_text(
            result["matched_skills"]
        )

        missing_text = self._missing_text(
            result["missing_skills"]
        )

        if score >= STRONG_MATCH_THRESHOLD:
            return (
                "Suggested next step: prioritised "
                "technical review. "
                f"The strongest detected alignment is in: "
                f"{matched_text}. "
                "The interview should validate practical "
                "depth, ownership and recent project evidence. "
                f"Areas to verify: {missing_text}."
            )

        if score >= GOOD_MATCH_THRESHOLD:
            return (
                "Suggested next step: standard technical "
                "interview. "
                f"Relevant alignment was detected in: "
                f"{matched_text}. "
                "Some requirements still need direct "
                "validation. "
                f"Topics to clarify: {missing_text}."
            )

        if score >= MEDIUM_MATCH_THRESHOLD:
            return (
                "Suggested next step: manual profile review "
                "before scheduling. "
                f"The profile has partial alignment in: "
                f"{matched_text}. "
                f"Important gaps or unclear areas include: "
                f"{missing_text}."
            )

        return (
            "Suggested next step: verify extraction quality "
            "and review the CV manually before deciding "
            "whether to continue. "
            f"Detected common points: {matched_text}. "
            f"Current gaps or unclear requirements: "
            f"{missing_text}."
        )

    def generate_interview_questions(
        self,
        matched_skills: List[str],
        missing_skills: List[str],
    ) -> List[str]:
        questions = []

        for skill in matched_skills[:3]:
            questions.append(
                f"Describe a concrete project where you "
                f"used {skill}. What was your personal "
                "contribution?"
            )

        for skill in missing_skills[:3]:
            questions.append(
                f"Do you have experience with {skill} or "
                "a comparable technology? Please give a "
                "specific example."
            )

        if not questions:
            questions = [
                (
                    "Which experience from your CV is "
                    "most relevant to this role?"
                ),
                (
                    "What was the most complex technical "
                    "problem you solved?"
                ),
                (
                    "Which required technology would you "
                    "need to learn first?"
                ),
            ]

        return questions

    def detect_red_flags(
        self,
        result: Dict[str, Any],
        cv_text: str,
    ) -> List[str]:
        signals = []

        if result["skill_score"] < 30:
            signals.append(
                "Low coverage of explicitly required "
                "job skills."
            )

        if result["semantic_score"] < 35:
            signals.append(
                "Low semantic alignment between the CV "
                "and job description."
            )

        if len(result["missing_skills"]) >= 5:
            signals.append(
                "Several required skills were not "
                "detected in the CV."
            )

        if len(str(cv_text).strip()) < 250:
            signals.append(
                "The extracted CV text is unusually short; "
                "document parsing or OCR quality should "
                "be checked."
            )

        if not signals:
            signals.append(
                "No major review signals were detected "
                "by the current rules."
            )

        return signals

    def verdict(
        self,
        score: float,
    ) -> str:
        if score >= STRONG_MATCH_THRESHOLD:
            return "Strong match"

        if score >= GOOD_MATCH_THRESHOLD:
            return "Good match"

        if score >= MEDIUM_MATCH_THRESHOLD:
            return "Partial match"

        return "Limited match"