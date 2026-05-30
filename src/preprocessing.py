import os
import re
import pandas as pd
from datasets import load_dataset
from skill_extraction import extract_skills

MAX_RESUMES = 10000


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " email_removed ", text)
    text = re.sub(r"\+?\d[\d\s().-]{7,}", " phone_removed ", text)
    text = re.sub(r"[^a-zA-Z0-9+#. ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    os.makedirs("data", exist_ok=True)

    resumes_dataset = load_dataset("datasetmaster/resumes", split="train")
    resumes_df = resumes_dataset.to_pandas()

    possible_cols = ["resume", "text", "content", "Resume", "resume_text"]
    resume_col = next((c for c in possible_cols if c in resumes_df.columns), resumes_df.columns[0])

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
        "Business analyst with Excel, communication, SQL and reporting experience"
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