import ast
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import mean_absolute_error, mean_squared_error

from baseline import baseline_hybrid_score


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"


def safe_list(value):
    try:
        return ast.literal_eval(value) if isinstance(value, str) else value
    except Exception:
        return []


def semantic_score(model, cv_text, job_text):
    cv_emb = model.encode(str(cv_text), convert_to_tensor=True, normalize_embeddings=True)
    job_emb = model.encode(str(job_text), convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(cv_emb, job_emb).item()


def optimize_weights():
    REPORTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_DIR / "training_pairs.csv")
    val_df = df[df["split"] == "val"].copy()

    model = SentenceTransformer(str(MODELS_DIR / "fine_tuned_sbert"))

    results = []

    for semantic_w in [0.4, 0.5, 0.6, 0.7]:
        for baseline_w in [0.1, 0.2, 0.3, 0.4]:
            skill_w = round(1 - semantic_w - baseline_w, 2)

            if skill_w < 0:
                continue

            y_true = []
            y_pred = []

            for _, row in val_df.iterrows():
                cv_text = row["cv_text"]
                job_text = row["job_text"]
                cv_skills = safe_list(row["cv_skills"])
                job_skills = safe_list(row["job_skills"])

                baseline = baseline_hybrid_score(cv_text, job_text, cv_skills, job_skills)
                sem = semantic_score(model, cv_text, job_text) * 100

                final_score = (
                    semantic_w * sem
                    + baseline_w * baseline["baseline_score"]
                    + skill_w * baseline["skill_score"]
                ) / 100

                y_true.append(float(row["label"]))
                y_pred.append(final_score)

            mae = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred) ** 0.5

            results.append({
                "semantic_weight": semantic_w,
                "baseline_weight": baseline_w,
                "skill_weight": skill_w,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
            })

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(REPORTS_DIR / "optimization_results.csv", index=False)

    print(results_df.head(10))
    print("Salvat: reports/optimization_results.csv")


if __name__ == "__main__":
    optimize_weights()