import ast
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MAX_PAIRS = 100000


def safe_list(value):
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    return []


def create_label(cv_skills, job_skills):
    cv_skills = set(cv_skills)
    job_skills = set(job_skills)

    if not job_skills:
        return 0.0

    return len(cv_skills & job_skills) / len(job_skills)


def create_training_pairs():
    resumes_path = DATA_DIR / "processed_resumes.csv"
    jobs_path = DATA_DIR / "processed_jobs.csv"
    output_path = DATA_DIR / "training_pairs.csv"

    resumes = pd.read_csv(resumes_path)
    jobs = pd.read_csv(jobs_path)

    rows = []

    for job_id, job in jobs.iterrows():
        job_text = str(job["clean_job"])
        job_skills = safe_list(job["required_skills"])

        for candidate_id, resume in resumes.iterrows():
            cv_text = str(resume["clean_resume"])
            cv_skills = safe_list(resume["skills_found"])

            label = create_label(cv_skills, job_skills)

            rows.append({
                "job_id": job_id,
                "candidate_id": candidate_id,
                "cv_text": cv_text,
                "job_text": job_text,
                "cv_skills": cv_skills,
                "job_skills": job_skills,
                "label": round(label, 4)
            })

            if len(rows) >= MAX_PAIRS:
                break

        if len(rows) >= MAX_PAIRS:
            break

    df = pd.DataFrame(rows)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_df.to_csv(output_path, index=False)

    print(f"Saved {len(final_df)} training pairs to {output_path}")


if __name__ == "__main__":
    create_training_pairs()