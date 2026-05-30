from pathlib import Path
from src.preprocessing import clean_text
from src.skill_extraction import extract_skills
from src.final_matcher import FinalMatcher


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

        if score >= 80:
            return "Candidat foarte potrivit. Recomandare: interviu tehnic."
        if score >= 60:
            return "Candidat promițător. Recomandare: interviu HR + verificare skill-uri lipsă."
        if score >= 40:
            return "Potrivire medie. Recomandare: rezervă sau analiză suplimentară."

        return "Potrivire scăzută. Recomandare: nu este prioritar."

    def generate_interview_questions(self, matched_skills, missing_skills):
        questions = []

        for skill in matched_skills[:3]:
            questions.append(f"Descrie un proiect în care ai folosit {skill}.")

        for skill in missing_skills[:3]:
            questions.append(f"Ai experiență cu {skill} sau tehnologii similare?")

        if not questions:
            questions.append("Ce experiență ai relevantă pentru acest rol?")

        return questions

    def detect_red_flags(self, result):
        flags = []

        if result["skill_score"] < 30:
            flags.append("Acoperire redusă a skill-urilor cerute.")
        if result["semantic_score"] < 40:
            flags.append("Similaritate semantică scăzută între CV și job.")
        if len(result["missing_skills"]) >= 5:
            flags.append("Multe skill-uri lipsă.")

        return flags or ["Nu au fost detectate red flags majore."]

    def run(self, cv_text, job_text):
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
            "cv_skills": cv_skills,
            "job_skills": job_skills,
            "recommendation": self.generate_recommendation(result),
            "interview_questions": self.generate_interview_questions(
                result["matched_skills"],
                result["missing_skills"]
            ),
            "red_flags": self.detect_red_flags(result)
        }