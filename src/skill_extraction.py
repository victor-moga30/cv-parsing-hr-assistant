import re

SKILLS = [
    "python", "java", "c++", "c#", "sql", "mysql", "postgresql",
    "excel", "power bi", "tableau", "machine learning", "deep learning",
    "nlp", "tensorflow", "pytorch", "scikit-learn", "html", "css",
    "javascript", "typescript", "react", "node.js", "spring",
    "git", "github", "docker", "kubernetes", "linux", "aws", "azure",
    "communication", "leadership", "teamwork", "problem solving",
    "data analysis", "statistics", "streamlit", "fastapi", "flask"
]

def extract_skills(text):
    text = str(text).lower()
    found = set()

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.add(skill)

    return sorted(found)

def matched_skills(cv_skills, job_skills):
    return sorted(list(set(cv_skills) & set(job_skills)))

def missing_skills(cv_skills, job_skills):
    return sorted(list(set(job_skills) - set(cv_skills)))