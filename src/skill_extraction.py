import re


SKILL_ALIASES = {
    "python": ["python", "python3"],
    "java": ["java", "core java"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "c sharp"],
    "sql": ["sql"],
    "mysql": ["mysql"],
    "postgresql": ["postgresql", "postgres"],
    "excel": ["excel", "microsoft excel"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural networks"],
    "nlp": ["nlp", "natural language processing"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch", "torch"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "react": ["react", "react.js", "reactjs"],
    "angular": ["angular"],
    "node.js": ["node.js", "nodejs", "node"],
    "spring": ["spring", "spring boot"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "git": ["git"],
    "github": ["github"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "linux": ["linux"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure"],
    "communication": ["communication", "communication skills"],
    "leadership": ["leadership"],
    "teamwork": ["teamwork", "team work"],
    "problem solving": ["problem solving", "problem-solving"],
    "data analysis": ["data analysis", "data analytics"],
    "statistics": ["statistics", "statistical analysis"],
    "streamlit": ["streamlit"],
    "algorithms": ["algorithms", "algorithm"],
    "data structures": ["data structures", "data structure"],
    "rest api": ["rest api", "restful api", "api"],
    "ci/cd": ["ci/cd", "continuous integration", "continuous deployment"],
    "reporting": ["reporting"],
    "testing": ["testing", "unit testing"],
    "oop": ["oop", "object oriented programming"],
    "agile": ["agile", "scrum"],
}


def _contains_alias(text: str, alias: str) -> bool:
    alias = alias.lower().strip()

    if alias in ["c++", "c#", "node.js", "ci/cd", "rest api"]:
        return alias in text

    pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
    return re.search(pattern, text) is not None


def extract_skills(text):
    text = str(text).lower()
    found = set()

    for canonical_skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            if _contains_alias(text, alias):
                found.add(canonical_skill)
                break

    return sorted(found)


def matched_skills(cv_skills, job_skills):
    return sorted(list(set(cv_skills) & set(job_skills)))


def missing_skills(cv_skills, job_skills):
    return sorted(list(set(job_skills) - set(cv_skills)))


def skill_coverage_percent(cv_skills, job_skills):
    job_skills = set(job_skills)
    cv_skills = set(cv_skills)

    if not job_skills:
        return 0.0

    return round(len(cv_skills & job_skills) / len(job_skills) * 100, 2)