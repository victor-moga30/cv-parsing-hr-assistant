import ast
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sentence_transformers import SentenceTransformer, util

from baseline import baseline_hybrid_score


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"


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


def semantic_score(model, cv_text, job_text):
    cv_emb = model.encode(str(cv_text), convert_to_tensor=True, normalize_embeddings=True)
    job_emb = model.encode(str(job_text), convert_to_tensor=True, normalize_embeddings=True)
    score = util.cos_sim(cv_emb, job_emb).item()
    return max(0, min(1, score))


def add_metrics(rows, name, y_true, preds):
    mse = mean_squared_error(y_true, preds)

    rows.append({
        "model": name,
        "MAE": round(mean_absolute_error(y_true, preds), 4),
        "RMSE": round(mse ** 0.5, 4),
        "Pearson": round(float(pearsonr(y_true, preds)[0]), 4),
        "Spearman": round(float(spearmanr(y_true, preds)[0]), 4),
    })


def evaluate():
    REPORTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_DIR / "training_pairs.csv")
    test_df = df[df["split"] == "test"].copy()

    y_true = test_df["label"].astype(float).tolist()

    pretrained_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    finetuned_path = MODELS_DIR / "fine_tuned_sbert"
    finetuned_model = SentenceTransformer(str(finetuned_path)) if finetuned_path.exists() else None

    baseline_preds = []
    pretrained_preds = []
    finetuned_preds = []

    for _, row in test_df.iterrows():
        cv_text = str(row["cv_text"])
        job_text = str(row["job_text"])

        cv_skills = safe_list(row["cv_skills"])
        job_skills = safe_list(row["job_skills"])

        baseline = baseline_hybrid_score(cv_text, job_text, cv_skills, job_skills)
        baseline_preds.append(baseline["baseline_score"] / 100)

        pretrained_preds.append(semantic_score(pretrained_model, cv_text, job_text))

        if finetuned_model:
            finetuned_preds.append(semantic_score(finetuned_model, cv_text, job_text))

    rows = []

    add_metrics(rows, "baseline_tfidf_skill", y_true, baseline_preds)
    add_metrics(rows, "pretrained_sbert", y_true, pretrained_preds)

    if finetuned_model:
        add_metrics(rows, "fine_tuned_sbert", y_true, finetuned_preds)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)

    print(metrics_df)
    print("Salvat: reports/model_metrics.csv")


if __name__ == "__main__":
    evaluate()