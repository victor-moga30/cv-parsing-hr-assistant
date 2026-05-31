from src.baseline import baseline_hybrid_score
from src.skill_extraction import matched_skills, missing_skills
from src.semantic_matcher import SemanticMatcher


class FinalMatcher:
    """
    Final hybrid matcher used by the HR assistant.

    The final score combines:
    - semantic similarity from Sentence-BERT
    - a classical baseline score
    - explicit skill coverage

    The weights below are selected from the optimization experiment
    saved in reports/optimization_results.csv.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.semantic_matcher = SemanticMatcher(model_name)

        self.semantic_weight = 0.40
        self.baseline_weight = 0.10
        self.skill_weight = 0.50

    def match(self, cv_text, job_text, cv_skills, job_skills):
        baseline = baseline_hybrid_score(cv_text, job_text, cv_skills, job_skills)
        semantic_score = self.semantic_matcher.similarity_score(cv_text, job_text)

        final_score = (
            self.semantic_weight * semantic_score
            + self.baseline_weight * baseline["baseline_score"]
            + self.skill_weight * baseline["skill_score"]
        )

        final_score = max(0.0, min(100.0, final_score))

        return {
            "final_score": round(final_score, 2),
            "semantic_score": round(semantic_score, 2),
            "baseline_score": round(baseline["baseline_score"], 2),
            "skill_score": round(baseline["skill_score"], 2),
            "tfidf_score": round(baseline["tfidf_score"], 2),
            "matched_skills": matched_skills(cv_skills, job_skills),
            "missing_skills": missing_skills(cv_skills, job_skills),
        }