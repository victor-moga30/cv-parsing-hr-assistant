from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.skill_extraction import skill_coverage_percent


def tfidf_similarity(cv_text, job_text):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w[\w+#.-]*\b"
    )

    matrix = vectorizer.fit_transform([str(cv_text), str(job_text)])
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

    return round(max(0.0, min(1.0, score)) * 100, 2)


def baseline_hybrid_score(cv_text, job_text, cv_skills, job_skills):
    tfidf_score = tfidf_similarity(cv_text, job_text)
    skill_score = skill_coverage_percent(cv_skills, job_skills)

    baseline_score = 0.4 * tfidf_score + 0.6 * skill_score
    baseline_score = max(0.0, min(100.0, baseline_score))

    return {
        "tfidf_score": round(tfidf_score, 2),
        "skill_score": round(skill_score, 2),
        "baseline_score": round(baseline_score, 2),
    }