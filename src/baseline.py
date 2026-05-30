from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def skill_match_score(cv_skills, job_skills):
    if not job_skills:
        return 0.0

    return round(len(set(cv_skills) & set(job_skills)) / len(set(job_skills)) * 100, 2)


def tfidf_similarity_score(cv_text, job_text):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    vectors = vectorizer.fit_transform([str(cv_text), str(job_text)])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]

    return round(score * 100, 2)


def baseline_hybrid_score(cv_text, job_text, cv_skills, job_skills):
    skill_score = skill_match_score(cv_skills, job_skills)
    tfidf_score = tfidf_similarity_score(cv_text, job_text)
    baseline_score = 0.65 * skill_score + 0.35 * tfidf_score

    return {
        "skill_score": round(skill_score, 2),
        "tfidf_score": round(tfidf_score, 2),
        "baseline_score": round(baseline_score, 2)
    }