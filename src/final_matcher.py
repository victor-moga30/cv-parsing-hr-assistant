from src.baseline import baseline_hybrid_score
from src.skill_extraction import matched_skills, missing_skills
from src.semantic_matcher import SemanticMatcher


class FinalMatcher:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.semantic_matcher = SemanticMatcher(model_name)

    def match(self, cv_text, job_text, cv_skills, job_skills):
        baseline = baseline_hybrid_score(cv_text, job_text, cv_skills, job_skills)
        semantic_score = self.semantic_matcher.similarity_score(cv_text, job_text)

        final_score = (
            0.55 * semantic_score
            + 0.25 * baseline["baseline_score"]
            + 0.20 * baseline["skill_score"]
        )

        return {
            "final_score": round(final_score, 2),
            "semantic_score": semantic_score,
            "baseline_score": baseline["baseline_score"],
            "skill_score": baseline["skill_score"],
            "tfidf_score": baseline["tfidf_score"],
            "matched_skills": matched_skills(cv_skills, job_skills),
            "missing_skills": missing_skills(cv_skills, job_skills),
        }