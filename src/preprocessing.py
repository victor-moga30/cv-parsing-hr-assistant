import ast
import json
import re
from pathlib import Path
from typing import Any, List

import pandas as pd

from src.skill_extraction import extract_skills


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RESUMES_PATH = DATA_DIR / "processed_resumes.csv"
JOBS_PATH = DATA_DIR / "processed_jobs.csv"

MAX_RESUMES = 10000


DEFAULT_JOB_DESCRIPTIONS = [
    (
        "We are looking for a Python developer with SQL, "
        "Git and machine learning knowledge."
    ),
    (
        "Data analyst role requiring Excel, Power BI, SQL "
        "and communication skills."
    ),
    (
        "Frontend developer with HTML, CSS, JavaScript, "
        "React and Git experience."
    ),
    (
        "Backend developer with Java, Spring, SQL and "
        "Docker experience."
    ),
    (
        "Machine learning engineer with Python, TensorFlow, "
        "deep learning and NLP skills."
    ),
    (
        "DevOps engineer with Docker, Kubernetes, Linux "
        "and CI/CD experience."
    ),
    (
        "Software engineer with C++, algorithms, data "
        "structures and problem solving skills."
    ),
    (
        "Business analyst with Excel, communication, SQL "
        "and reporting experience."
    ),
]


SENSITIVE_OR_CONTACT_KEYS = {
    "name",
    "email",
    "phone",
    "telephone",
    "linkedin",
    "github",
    "photo",
    "image",
    "age",
    "gender",
    "ethnicity",
    "address",
    "location",
    "city",
    "country",
    "remote_preference",
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

LABELED_CONTACT_LINE_PATTERN = re.compile(
    r"(?im)^\s*"
    r"(?:email|e-mail|phone|mobile|telephone|tel|linkedin|github)"
    r"\s*[:\-]\s*.*$"
)

INTERNATIONAL_PHONE_PATTERN = re.compile(
    r"(?<!\w)"
    r"\+\d[\d\s().-]{7,}\d"
    r"(?!\w)"
)

PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:unknown|not provided|not available|n/?a|null|none)\b",
    flags=re.IGNORECASE,
)


def fix_spaced_letters(text: str) -> str:
    text = str(text)

    pattern = (
        r"\b(?:[A-Za-z]\s+)"
        r"{2,}[A-Za-z]\b"
    )

    def join_match(match):
        return match.group(0).replace(
            " ",
            "",
        )

    previous = None

    while previous != text:
        previous = text

        text = re.sub(
            pattern,
            join_match,
            text,
        )

    return text


def _flatten_relevant_json_values(
    value: Any,
    parent_key: str = "",
) -> List[str]:
    """
    Flatten useful CV fields while excluding contact and
    explicit sensitive/proxy fields.
    """
    if isinstance(value, dict):
        collected = []

        for key, child in value.items():
            normalized_key = str(
                key
            ).strip().lower()

            if normalized_key in SENSITIVE_OR_CONTACT_KEYS:
                continue

            collected.extend(
                _flatten_relevant_json_values(
                    child,
                    parent_key=normalized_key,
                )
            )

        return collected

    if isinstance(value, list):
        collected = []

        for child in value:
            collected.extend(
                _flatten_relevant_json_values(
                    child,
                    parent_key=parent_key,
                )
            )

        return collected

    if isinstance(
        value,
        (str, int, float),
    ):
        text = str(value).strip()

        if (
            text
            and not PLACEHOLDER_PATTERN.fullmatch(text)
        ):
            return [text]

    return []


def _extract_relevant_text(text: str) -> str:
    raw_text = str(text).strip()

    if not raw_text:
        return ""

    if (
        raw_text.startswith("{")
        or raw_text.startswith("[")
    ):
        try:
            parsed = json.loads(
                raw_text
            )

            values = _flatten_relevant_json_values(
                parsed
            )

            if values:
                return " ".join(values)

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            try:
                parsed = ast.literal_eval(
                    raw_text
                )

                values = _flatten_relevant_json_values(
                    parsed
                )

                if values:
                    return " ".join(values)

            except (
                ValueError,
                SyntaxError,
                TypeError,
            ):
                pass

    return raw_text


def clean_text(text: str) -> str:
    """
    Prepare text for semantic, lexical and skill matching.

    Applicant contact information is extracted separately
    before this function is called.
    """
    text = _extract_relevant_text(text)
    text = fix_spaced_letters(text)

    text = text.replace(
        "\x00",
        " ",
    )

    text = text.replace(
        "C + +",
        "C++",
    )

    text = text.replace(
        "c + +",
        "c++",
    )

    text = text.replace(
        "C / C + +",
        "C/C++",
    )

    text = text.replace(
        "c / c + +",
        "c/c++",
    )

    text = re.sub(
        r"([A-Za-z])\s*\+\s*\+",
        r"\1++",
        text,
    )

    text = LABELED_CONTACT_LINE_PATTERN.sub(
        " ",
        text,
    )

    text = EMAIL_PATTERN.sub(
        " ",
        text,
    )

    text = URL_PATTERN.sub(
        " ",
        text,
    )

    text = INTERNATIONAL_PHONE_PATTERN.sub(
        " ",
        text,
    )

    text = PLACEHOLDER_PATTERN.sub(
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _download_resume_source() -> pd.DataFrame:
    from datasets import load_dataset

    dataset = load_dataset(
        "datasetmaster/resumes",
        split="train",
    )

    resumes_df = dataset.to_pandas()

    possible_columns = [
        "resume",
        "text",
        "content",
        "Resume",
        "resume_text",
    ]

    resume_column = next(
        (
            column
            for column in possible_columns
            if column in resumes_df.columns
        ),
        resumes_df.columns[0],
    )

    return (
        resumes_df[[resume_column]]
        .rename(
            columns={
                resume_column: "resume_text"
            }
        )
        .dropna()
        .head(MAX_RESUMES)
        .copy()
    )


def _load_resume_source() -> pd.DataFrame:
    """
    Reuse the original resume_text column when the
    processed file already exists.
    """
    if RESUMES_PATH.is_file():
        existing = pd.read_csv(
            RESUMES_PATH
        )

        if "resume_text" in existing.columns:
            return (
                existing[["resume_text"]]
                .dropna()
                .head(MAX_RESUMES)
                .copy()
            )

    return _download_resume_source()


def _load_job_source() -> pd.DataFrame:
    if JOBS_PATH.is_file():
        existing = pd.read_csv(
            JOBS_PATH
        )

        if "job_description" in existing.columns:
            return (
                existing[["job_description"]]
                .dropna()
                .copy()
            )

    return pd.DataFrame({
        "job_description": DEFAULT_JOB_DESCRIPTIONS
    })


def main() -> None:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    resumes_df = _load_resume_source()
    jobs_df = _load_job_source()

    resumes_df["clean_resume"] = (
        resumes_df["resume_text"]
        .apply(clean_text)
    )

    resumes_df = resumes_df[
        resumes_df["clean_resume"].str.len() > 0
    ].copy()

    resumes_df["skills_found"] = (
        resumes_df["clean_resume"]
        .apply(extract_skills)
    )

    resumes_df["num_skills"] = (
        resumes_df["skills_found"]
        .apply(len)
    )

    jobs_df["clean_job"] = (
        jobs_df["job_description"]
        .apply(clean_text)
    )

    jobs_df = jobs_df[
        jobs_df["clean_job"].str.len() > 0
    ].copy()

    jobs_df["required_skills"] = (
        jobs_df["clean_job"]
        .apply(extract_skills)
    )

    resumes_df.to_csv(
        RESUMES_PATH,
        index=False,
    )

    jobs_df.to_csv(
        JOBS_PATH,
        index=False,
    )

    print(
        f"Saved {len(resumes_df)} resumes "
        f"to {RESUMES_PATH}"
    )

    print(
        f"Saved {len(jobs_df)} jobs "
        f"to {JOBS_PATH}"
    )

    print(
        "Average detected resume skills: "
        f"{resumes_df['num_skills'].mean():.2f}"
    )


if __name__ == "__main__":
    main()