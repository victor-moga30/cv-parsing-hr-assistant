import os
import re

import pandas as pd

from src.skill_extraction import extract_skills


MAX_RESUMES = 10000


def fix_spaced_letters(text: str) -> str:
    pattern = r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b"

    def join_match(match):
        return match.group(0).replace(" ", "")

    previous = None
    while previous != text:
        previous = text
        text = re.sub(pattern, join_match, text)

    return text


def clean_text(text):
    text = str(text)
    text = fix_spaced_letters(text)

    text = text.replace("\x00", " ")
    text = text.replace("C + +", "C++")
    text = text.replace("c + +", "c++")
    text = text.replace("C / C + +", "C/C++")
    text = text.replace("c / c + +", "c/c++")

    text = re.sub(r"([A-Za-z])\s*\+\s*\+", r"\1++", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():
    from datasets import load_dataset

    os.makedirs("data", exist_ok=True)

    resumes_dataset = load_dataset("datasetmaster/resumes", split="train")
    resumes_df = resumes_dataset.to_pandas()

    possible_cols = ["resume", "text", "content", "Resume", "resume_text"]
    resume_col = next((col for col in possible_cols if col in resumes_df.columns), resumes_df.columns[0])

    resumes_df = resumes_df[[resume_col]].rename(columns={resume_col: "resume_text"})
    resumes_df = resumes_df.dropna().head(MAX_RESUMES)

    resumes_df["clean_resume"] = resumes_df["resume_text"].apply(clean_text)
    resumes_df["skills_found"] = resumes_df["clean_resume"].apply(extract_skills)
    resumes_df["num_skills"] = resumes_df["skills_found"].apply(len)

    job_descriptions = [
        "We are looking for a Python developer with SQL, Git and machine learning knowledge.",
        "Data analyst role requiring Excel, Power BI, SQL and communication skills.",
        "Frontend developer with HTML, CSS, JavaScript, React and Git experience.",
        "Backend developer with Java, Spring, SQL and Docker experience.",
        "Machine learning engineer with Python, TensorFlow, deep learning and NLP skills.",
        "DevOps engineer with Docker, Kubernetes, Linux and CI/CD experience.",
        "Software engineer with C++, algorithms, data structures and problem solving skills.",
        "Business analyst with Excel, communication, SQL and reporting experience",
    ]

    jobs_df = pd.DataFrame({"job_description": job_descriptions})
    jobs_df["clean_job"] = jobs_df["job_description"].apply(clean_text)
    jobs_df["required_skills"] = jobs_df["clean_job"].apply(extract_skills)

    resumes_df.to_csv("data/processed_resumes.csv", index=False)
    jobs_df.to_csv("data/processed_jobs.csv", index=False)

    print("Saved data/processed_resumes.csv")
    print("Saved data/processed_jobs.csv")


if __name__ == "__main__":
    main()