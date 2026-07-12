import re
from typing import Iterable, List


SKILLS = {
    "python": ["python"],
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
    "go": ["golang", "go language"],
    "kotlin": ["kotlin"],
    "swift": ["swift"],

    "sql": ["sql"],
    "mysql": ["mysql", "my sql"],
    "postgresql": [
        "postgresql",
        "postgres",
        "postgre sql",
        "postresql",
    ],
    "sqlite": ["sqlite", "sqlite3"],
    "mongodb": ["mongodb", "mongo db"],
    "oracle": ["oracle database", "oracle db"],
    "database": ["database", "databases", "dbms", "sgbd"],

    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "artificial intelligence": [
        "artificial intelligence",
        "ai",
    ],
    "nlp": ["nlp", "natural language processing"],
    "data science": ["data science", "data scientist"],
    "data analysis": [
        "data analysis",
        "data analyst",
        "analytics",
    ],
    "computer vision": ["computer vision", "opencv"],
    "statistics": ["statistics", "statistical analysis"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": [
        "scikit-learn",
        "sklearn",
        "scikit learn",
    ],
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
    "bash": [
        "bash",
        "shell scripting",
        "bash scripting",
    ],
    "ci/cd": [
        "ci/cd",
        "ci cd",
        "cicd",
        "continuous integration",
        "continuous deployment",
    ],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud"],

    "oop": [
        "oop",
        "object oriented programming",
        "object-oriented programming",
    ],
    "data structures": [
        "data structures",
        "data structure",
    ],
    "algorithms": ["algorithms", "algorithm"],
    "testing": [
        "testing",
        "unit testing",
        "unit tests",
        "testare",
        "debugging",
    ],
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
    "rest api": [
        "rest api",
        "rest apis",
        "restful api",
        "restful apis",
        "web api",
        "web apis",
    ],

    "agile": ["agile", "scrum", "kanban"],
    "teamwork": [
        "teamwork",
        "team work",
        "collaboration",
    ],
    "communication": ["communication", "comunicare"],
    "problem solving": [
        "problem solving",
        "problem-solving",
    ],
    "leadership": ["leadership"],
    "project management": [
        "project management",
        "management de proiect",
    ],
}


EMAIL_PATTERN = re.compile(
    r"\b[\w.+-]+@[\w.-]+\.\w+\b",
    flags=re.IGNORECASE,
)

URL_PATTERN = re.compile(
    r"\b(?:https?://|www\.)\S+"
    r"|\b(?:linkedin\.com|github\.com)/\S+",
    flags=re.IGNORECASE,
)

PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:github|git hub|linkedin|email|phone|name)\s+"
    r"(?:unknown|not provided|not available|n/?a)\b",
    flags=re.IGNORECASE,
)


def fix_spaced_letters(text: str) -> str:
    """
    Join OCR artefacts such as:
        P y t h o n
    into:
        Python
    """
    text = str(text)
    pattern = r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b"

    def join_match(match):
        return match.group(0).replace(" ", "")

    previous = None

    while previous != text:
        previous = text
        text = re.sub(
            pattern,
            join_match,
            text,
        )

    return text


def _remove_non_skill_metadata(text: str) -> str:
    """
    Remove contact information before skill detection.

    A GitHub profile URL is contact information and should not
    automatically count as proof of GitHub proficiency.
    """
    text = EMAIL_PATTERN.sub(
        " ",
        str(text),
    )

    text = URL_PATTERN.sub(
        " ",
        text,
    )

    text = PLACEHOLDER_PATTERN.sub(
        " ",
        text,
    )

    return text


def normalize_text(text: str) -> str:
    text = _remove_non_skill_metadata(text)
    text = fix_spaced_letters(text).lower()

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
        text = text.replace(
            old,
            new,
        )

    text = re.sub(
        r"([a-z])\s*\+\s*\+",
        r"\1++",
        text,
    )

    for character in [
        "|",
        ",",
        ";",
        ":",
        "\n",
        "\t",
        "•",
        "●",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
    ]:
        text = text.replace(
            character,
            " ",
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return f" {text.strip()} "


def _alias_pattern(alias: str) -> str:
    """
    Create boundary-safe matching.

    This prevents:
    - Java matching JavaScript;
    - AI matching email;
    - C matching C++ or C# through the one-letter alias.
    """
    escaped_alias = re.escape(alias)

    if alias == "c":
        return (
            rf"(?<![a-z0-9])"
            rf"{escaped_alias}"
            rf"(?![a-z0-9+#/])"
        )

    return (
        rf"(?<![a-z0-9])"
        rf"{escaped_alias}"
        rf"(?![a-z0-9])"
    )


def extract_skills(text: str) -> List[str]:
    normal_text = normalize_text(text)
    found = set()

    for skill, aliases in SKILLS.items():
        for alias in aliases:
            normalized_alias = normalize_text(
                alias
            ).strip()

            if not normalized_alias:
                continue

            pattern = _alias_pattern(
                normalized_alias
            )

            if re.search(
                pattern,
                normal_text,
            ):
                found.add(skill)
                break

    return sorted(found)


def matched_skills(
    cv_skills: Iterable[str],
    job_skills: Iterable[str],
) -> List[str]:
    return sorted(
        set(cv_skills)
        & set(job_skills)
    )


def missing_skills(
    cv_skills: Iterable[str],
    job_skills: Iterable[str],
) -> List[str]:
    return sorted(
        set(job_skills)
        - set(cv_skills)
    )


def skill_coverage_percent(
    cv_skills: Iterable[str],
    job_skills: Iterable[str],
) -> float:
    required = set(job_skills)
    candidate = set(cv_skills)

    if not required:
        return 0.0

    coverage = (
        len(candidate & required)
        / len(required)
        * 100
    )

    return round(
        coverage,
        2,
    )