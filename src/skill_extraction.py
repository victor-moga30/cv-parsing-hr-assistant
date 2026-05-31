import re


SKILLS = {
    "python": ["python", "py"],
    "java": ["java"],
    "c": ["c", "c language", "c/c++"],
    "c++": ["c++", "cpp", "c/c++"],
    "c#": ["c#", "c sharp", "csharp"],
    "javascript": ["javascript", "java script", "js"],
    "typescript": ["typescript", "type script", "ts"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "php": ["php"],
    "ruby": ["ruby"],
    "go": ["go", "golang"],
    "kotlin": ["kotlin"],
    "swift": ["swift"],

    "sql": ["sql"],
    "mysql": ["mysql", "my sql"],
    "postgresql": ["postgresql", "postgres", "postgre sql", "postresql"],
    "sqlite": ["sqlite", "sqlite3"],
    "mongodb": ["mongodb", "mongo db"],
    "oracle": ["oracle database", "oracle db"],
    "database": ["database", "databases", "dbms", "sgbd"],

    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "nlp": ["nlp", "natural language processing"],
    "data science": ["data science", "data scientist"],
    "data analysis": ["data analysis", "data analyst", "analytics"],
    "computer vision": ["computer vision", "cv"],
    "statistics": ["statistics", "statistical analysis"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "tensorflow": ["tensorflow", "tensor flow"],
    "pytorch": ["pytorch", "py torch"],
    "keras": ["keras"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "plotly": ["plotly"],
    "power bi": ["power bi", "powerbi"],
    "excel": ["excel", "microsoft excel"],
    "tableau": ["tableau"],

    "git": ["git"],
    "github": ["github", "git hub"],
    "gitlab": ["gitlab", "git lab"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "linux": ["linux"],
    "bash": ["bash", "shell", "shell scripting", "bash scripting"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud"],

    "oop": ["oop", "object oriented programming", "object-oriented programming"],
    "data structures": ["data structures", "data structure"],
    "algorithms": ["algorithms", "algorithm"],
    "testing": ["testing", "unit testing", "unit tests", "testare", "debugging", "debug"],
    "junit": ["junit"],
    "pytest": ["pytest"],
    "selenium": ["selenium"],

    "react": ["react", "reactjs", "react.js"],
    "angular": ["angular", "angularjs"],
    "vue": ["vue", "vuejs", "vue.js"],
    "node.js": ["node.js", "nodejs", "node js"],
    "express": ["express", "express.js", "expressjs"],
    "spring": ["spring", "spring boot", "springboot"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi", "fast api"],
    "streamlit": ["streamlit"],
    "qt": ["qt"],
    "javafx": ["javafx", "java fx"],
    "jdbc": ["jdbc"],
    "hibernate": ["hibernate"],
    "rest api": ["rest api", "rest", "api", "web api"],

    "agile": ["agile", "scrum", "kanban"],
    "teamwork": ["teamwork", "team work", "collaboration"],
    "communication": ["communication", "comunicare"],
    "problem solving": ["problem solving", "problem-solving"],
    "leadership": ["leadership"],
    "project management": ["project management", "management de proiect"],
}


def fix_spaced_letters(text: str) -> str:
    text = str(text)
    pattern = r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b"

    def join_match(match):
        return match.group(0).replace(" ", "")

    previous = None
    while previous != text:
        previous = text
        text = re.sub(pattern, join_match, text)

    return text


def normalize_text(text: str) -> str:
    text = fix_spaced_letters(str(text)).lower()

    fixes = {
        "c + +": "c++",
        "c ++": "c++",
        "c+ +": "c++",
        "c / c++": "c/c++",
        "c/c++": " c/c++ ",
        "c #": "c#",
        "java fx": "javafx",
        "node js": "node.js",
        "react js": "reactjs",
        "vue js": "vuejs",
        "postgre sql": "postgresql",
        "postre sql": "postgresql",
        "scikit learn": "scikit-learn",
        "powerbi": "power bi",
        "tensor flow": "tensorflow",
        "py torch": "pytorch",
        "object-oriented": "object oriented",
        "problem-solving": "problem solving",
    }

    for old, new in fixes.items():
        text = text.replace(old, new)

    text = re.sub(r"([a-z])\s*\+\s*\+", r"\1++", text)

    for ch in ["|", ",", ";", ":", "\n", "\t", "•", "●", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, " ")

    text = re.sub(r"\s+", " ", text)
    return f" {text.strip()} "


def compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9+#/.-]", "", normalize_text(text))


def extract_skills(text: str):
    normal = normalize_text(text)
    compact = compact_text(text)

    found = set()

    for skill, aliases in SKILLS.items():
        for alias in aliases:
            alias_normal = normalize_text(alias).strip()
            alias_compact = compact_text(alias)

            pattern = r"(?<![a-z0-9+#.-])" + re.escape(alias_normal) + r"(?![a-z0-9+#.-])"

            if re.search(pattern, normal) or alias_compact in compact:
                found.add(skill)
                break

    return sorted(found)


def matched_skills(cv_skills, job_skills):
    return sorted(set(cv_skills) & set(job_skills))


def missing_skills(cv_skills, job_skills):
    return sorted(set(job_skills) - set(cv_skills))


def skill_coverage_percent(cv_skills, job_skills):
    job_skills = set(job_skills)
    cv_skills = set(cv_skills)

    if not job_skills:
        return 0.0

    return round(len(cv_skills & job_skills) / len(job_skills) * 100, 2)