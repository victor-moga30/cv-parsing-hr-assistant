from pathlib import Path

from src.preprocessing import clean_text
from src.skill_extraction import extract_skills
from src.final_matcher import FinalMatcher
from src.document_parser import extract_applicant_info


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fine_tuned_sbert"


class HRAssistantAgent:
    def __init__(self):
        if MODEL_PATH.exists():
            self.matcher = FinalMatcher(model_name=str(MODEL_PATH))
        else:
            self.matcher = FinalMatcher()

    def generate_recommendation(self, result):
        score = result["final_score"]
        missing = result["missing_skills"]

        if score >= 80:
            return (
                "Candidat foarte potrivit. Recomandare: merge direct la interviu tehnic. "
                "CV-ul acopera foarte bine cerintele jobului."
            )

        if score >= 60:
            return (
                "Candidat bun. Recomandare: interviu HR si apoi verificare tehnica pe skill-urile lipsa: "
                + (", ".join(missing[:4]) if missing else "nu exista skill-uri importante lipsa.")
            )

        if score >= 40:
            return (
                "Candidat cu potrivire medie. Recomandare: poate fi pastrat ca rezerva sau evaluat "
                "daca experienta generala este relevanta."
            )

        return (
            "Candidat slab potrivit pentru acest job. Recomandare: nu este prioritar pentru rolul curent, "
            "dar poate fi potrivit pentru alta pozitie."
        )

    def generate_interview_questions(self, matched_skills, missing_skills):
        questions = []

        for skill in matched_skills[:3]:
            questions.append(f"Descrie un proiect concret in care ai folosit {skill}.")

        for skill in missing_skills[:3]:
            questions.append(f"Ai experienta cu {skill} sau cu o tehnologie similara? Da un exemplu.")

        if not questions:
            questions.append("Ce experienta ai care este relevanta pentru acest rol?")
            questions.append("Care a fost cel mai complex proiect tehnic la care ai lucrat?")
            questions.append("Ce tehnologii ai invata rapid pentru a te adapta la acest job?")

        return questions

    def detect_red_flags(self, result, cv_text):
        flags = []

        if result["skill_score"] < 30:
            flags.append("Acoperire redusa a skill-urilor cerute.")

        if result["semantic_score"] < 35:
            flags.append("Similaritate semantica scazuta intre CV si descrierea jobului.")

        if len(result["missing_skills"]) >= 5:
            flags.append("Multe skill-uri importante lipsesc din CV.")

        if len(str(cv_text).strip()) < 250:
            flags.append("Textul extras din CV este foarte scurt. PDF-ul poate fi scanat sau greu de citit.")

        return flags or ["Nu au fost detectate red flags majore."]

    def verdict(self, score):
        if score >= 80:
            return "Strong match"
        if score >= 60:
            return "Good match"
        if score >= 40:
            return "Medium match"
        return "Weak match"

    def run(self, cv_text, job_text):
        applicant_info = extract_applicant_info(cv_text)

        clean_cv = clean_text(cv_text)
        clean_job = clean_text(job_text)

        cv_skills = extract_skills(clean_cv)
        job_skills = extract_skills(clean_job)

        result = self.matcher.match(
            clean_cv,
            clean_job,
            cv_skills,
            job_skills
        )

        return {
            **result,
            "applicant_info": applicant_info,
            "cv_skills": cv_skills,
            "job_skills": job_skills,
            "recommendation": self.generate_recommendation(result),
            "interview_questions": self.generate_interview_questions(
                result["matched_skills"],
                result["missing_skills"]
            ),
            "red_flags": self.detect_red_flags(result, cv_text),
            "verdict": self.verdict(result["final_score"]),
        }