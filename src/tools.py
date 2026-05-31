from typing import Any, Dict, List

from src.document_parser import extract_applicant_info, extract_text_from_uploaded_file
from src.preprocessing import clean_text
from src.skill_extraction import extract_skills
from src.final_matcher import FinalMatcher


class DocumentParserTool:
    """
    Tool used for extracting text from uploaded CV files.

    It supports:
    - PDF files
    - TXT files
    - images, if OCR is available
    """

    def run(self, uploaded_file) -> str:
        if uploaded_file is None:
            return ""

        return extract_text_from_uploaded_file(uploaded_file)


class ApplicantInfoTool:
    """
    Tool used for extracting basic applicant information.

    This information is shown to the recruiter, but it is not used directly
    in the matching score, in order to reduce possible bias.
    """

    def run(self, cv_text: str) -> Dict[str, Any]:
        return extract_applicant_info(cv_text)


class TextCleaningTool:
    """
    Tool used for normalizing text extracted from CVs and job descriptions.
    """

    def run(self, text: str) -> str:
        return clean_text(text)


class SkillExtractionTool:
    """
    Tool used for extracting technical and soft skills from text.
    """

    def run(self, text: str) -> List[str]:
        return extract_skills(text)


class MatchingTool:
    """
    Tool used for computing the final matching result between a CV and a job.
    """

    def __init__(self, matcher: FinalMatcher):
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
    Tool used for generating HR explanations, interview questions,
    red flags and a simple final verdict.
    """

    def generate_recommendation(self, result: Dict[str, Any]) -> str:
        score = result["final_score"]
        matched = result["matched_skills"]
        missing = result["missing_skills"]

        matched_text = ", ".join(matched[:8]) if matched else "no clear common technical skills"
        missing_text = ", ".join(missing[:5]) if missing else "no major missing skills"

        if score >= 75:
            return (
                "The candidate is a very good option for this role. "
                f"The profile matches the technical requirements well, especially on: {matched_text}. "
                "I recommend a priority technical interview, focused on practical projects and real autonomy. "
                f"Areas to verify: {missing_text}."
            )

        if score >= 60:
            return (
                "The candidate is suitable for a junior or internship role. "
                f"The CV shows relevant technical background on: {matched_text}. "
                "However, not all requirements are fully covered. "
                "I recommend a short technical interview with practical questions. "
                f"Topics to clarify: {missing_text}."
            )

        if score >= 45:
            return (
                "The candidate has potential, but the match is incomplete. "
                f"There are some common points, such as: {matched_text}. "
                f"Important missing elements: {missing_text}. "
                "The candidate can be kept as a backup option or considered for a more entry-level role."
            )

        return (
            "The candidate has a low match for this job. "
            f"There are only a few common points, such as: {matched_text}. "
            f"The differences from the job requirements are significant: {missing_text}. "
            "I do not recommend prioritizing this candidate for the current role."
        )

    def generate_interview_questions(
        self,
        matched_skills: List[str],
        missing_skills: List[str],
    ) -> List[str]:
        questions = []

        for skill in matched_skills[:3]:
            questions.append(f"Describe a concrete project where you used {skill}.")

        for skill in missing_skills[:3]:
            questions.append(f"Do you have experience with {skill} or a similar technology? Give an example.")

        if not questions:
            questions.append("What experience do you have that is relevant for this role?")
            questions.append("What was the most complex technical project you worked on?")
            questions.append("What technologies would you learn quickly in order to adapt to this job?")

        return questions

    def detect_red_flags(self, result: Dict[str, Any], cv_text: str) -> List[str]:
        flags = []

        if result["skill_score"] < 30:
            flags.append("Low coverage of the required job skills.")

        if result["semantic_score"] < 35:
            flags.append("Low semantic similarity between the CV and the job description.")

        if len(result["missing_skills"]) >= 5:
            flags.append("Many important skills from the job description are missing from the CV.")

        if len(str(cv_text).strip()) < 250:
            flags.append("The extracted CV text is very short. The PDF may be scanned or hard to read.")

        if not flags:
            flags.append("No major red flags were detected.")

        return flags

    def verdict(self, score: float) -> str:
        if score >= 80:
            return "Strong match"
        if score >= 60:
            return "Good match"
        if score >= 40:
            return "Medium match"

        return "Weak match"