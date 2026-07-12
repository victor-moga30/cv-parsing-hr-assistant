import ast
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

MAX_PAIRS = 100000
RANDOM_SEED = 42


def safe_list(value) -> List[str]:
    if isinstance(value, list):
        return value

    if isinstance(
        value,
        (tuple, set),
    ):
        return list(value)

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(
                value
            )

            if isinstance(
                parsed,
                (list, tuple, set),
            ):
                return list(parsed)

        except (
            ValueError,
            SyntaxError,
        ):
            pass

    return []


def create_label(
    cv_skills,
    job_skills,
) -> float:
    candidate_skills = set(
        cv_skills
    )

    required_skills = set(
        job_skills
    )

    if not required_skills:
        return 0.0

    return (
        len(
            candidate_skills
            & required_skills
        )
        / len(required_skills)
    )


def _validate_input_data(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
) -> None:
    required_resume_columns = {
        "clean_resume",
        "skills_found",
    }

    required_job_columns = {
        "clean_job",
        "required_skills",
    }

    missing_resume_columns = (
        required_resume_columns
        .difference(resumes.columns)
    )

    missing_job_columns = (
        required_job_columns
        .difference(jobs.columns)
    )

    if missing_resume_columns:
        raise ValueError(
            "processed_resumes.csv is missing columns: "
            + ", ".join(
                sorted(missing_resume_columns)
            )
        )

    if missing_job_columns:
        raise ValueError(
            "processed_jobs.csv is missing columns: "
            + ", ".join(
                sorted(missing_job_columns)
            )
        )

    if len(resumes) < 3:
        raise ValueError(
            "At least three resumes are required "
            "for grouped splitting."
        )

    if jobs.empty:
        raise ValueError(
            "At least one job description is required."
        )


def _candidate_group_split(
    candidate_ids,
):
    """
    Split by candidate instead of individual CV-job pairs.

    This guarantees that the same CV cannot appear in
    training and testing with different job descriptions.
    """
    train_ids, temporary_ids = train_test_split(
        list(candidate_ids),
        test_size=0.30,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    validation_ids, test_ids = train_test_split(
        temporary_ids,
        test_size=0.50,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    return (
        set(train_ids),
        set(validation_ids),
        set(test_ids),
    )


def create_training_pairs() -> None:
    resumes_path = (
        DATA_DIR
        / "processed_resumes.csv"
    )

    jobs_path = (
        DATA_DIR
        / "processed_jobs.csv"
    )

    output_path = (
        DATA_DIR
        / "training_pairs.csv"
    )

    if not resumes_path.is_file():
        raise FileNotFoundError(
            f"Missing {resumes_path}. "
            "Run: python -m src.preprocessing"
        )

    if not jobs_path.is_file():
        raise FileNotFoundError(
            f"Missing {jobs_path}. "
            "Run: python -m src.preprocessing"
        )

    resumes = (
        pd.read_csv(resumes_path)
        .reset_index(drop=True)
    )

    jobs = (
        pd.read_csv(jobs_path)
        .reset_index(drop=True)
    )

    _validate_input_data(
        resumes,
        jobs,
    )

    (
        train_ids,
        validation_ids,
        test_ids,
    ) = _candidate_group_split(
        resumes.index
    )

    rows = []

    for job_id, job in jobs.iterrows():
        job_text = str(
            job["clean_job"]
        )

        job_skills = safe_list(
            job["required_skills"]
        )

        for (
            candidate_id,
            resume,
        ) in resumes.iterrows():
            cv_text = str(
                resume["clean_resume"]
            )

            cv_skills = safe_list(
                resume["skills_found"]
            )

            if candidate_id in train_ids:
                split = "train"

            elif candidate_id in validation_ids:
                split = "val"

            elif candidate_id in test_ids:
                split = "test"

            else:
                raise RuntimeError(
                    f"Candidate {candidate_id} "
                    "was not assigned to a split."
                )

            rows.append({
                "job_id": int(job_id),
                "candidate_id": int(candidate_id),
                "cv_text": cv_text,
                "job_text": job_text,
                "cv_skills": cv_skills,
                "job_skills": job_skills,
                "label": round(
                    create_label(
                        cv_skills,
                        job_skills,
                    ),
                    4,
                ),
                "split": split,
            })

            if len(rows) >= MAX_PAIRS:
                break

        if len(rows) >= MAX_PAIRS:
            break

    final_df = pd.DataFrame(
        rows
    )

    if final_df.empty:
        raise RuntimeError(
            "No training pairs were generated."
        )

    final_df = final_df.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    train_candidates = set(
        final_df.loc[
            final_df["split"] == "train",
            "candidate_id",
        ]
    )

    validation_candidates = set(
        final_df.loc[
            final_df["split"] == "val",
            "candidate_id",
        ]
    )

    test_candidates = set(
        final_df.loc[
            final_df["split"] == "test",
            "candidate_id",
        ]
    )

    if (
        train_candidates
        & validation_candidates
    ):
        raise RuntimeError(
            "Candidate leakage detected between "
            "train and validation."
        )

    if (
        train_candidates
        & test_candidates
    ):
        raise RuntimeError(
            "Candidate leakage detected between "
            "train and test."
        )

    if (
        validation_candidates
        & test_candidates
    ):
        raise RuntimeError(
            "Candidate leakage detected between "
            "validation and test."
        )

    final_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved {len(final_df)} training pairs "
        f"to {output_path}"
    )

    print("Pair distribution:")

    print(
        final_df["split"]
        .value_counts()
        .to_string()
    )

    print("Candidate distribution:")

    print(
        pd.Series({
            "train": len(train_candidates),
            "val": len(validation_candidates),
            "test": len(test_candidates),
        }).to_string()
    )

    print(
        "Candidate overlap across splits: 0"
    )


if __name__ == "__main__":
    create_training_pairs()