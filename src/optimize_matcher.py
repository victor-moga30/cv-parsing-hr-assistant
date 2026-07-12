import ast
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.baseline import baseline_hybrid_score
from src.semantic_matcher import SemanticMatcher


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

TRAINING_PAIRS_PATH = DATA_DIR / "training_pairs.csv"
MODEL_PATH = MODELS_DIR / "fine_tuned_sbert"
OPTIMIZATION_REPORT_PATH = REPORTS_DIR / "optimization_results.csv"


# This is intentionally a constrained grid search.
#
# The training labels are generated from skill coverage. Therefore, a fully
# unrestricted optimisation would trivially prefer a skill-only score:
#
# semantic_weight = 0
# baseline_weight = 0
# skill_weight = 1
#
# These constraints preserve the intended hybrid architecture while still
# selecting the best combination on the validation set.
SEMANTIC_WEIGHTS = [0.40, 0.50, 0.60, 0.70]
BASELINE_WEIGHTS = [0.10, 0.20, 0.30, 0.40]


def safe_list(value):
    """
    Convert a value loaded from the CSV into a Python list.

    The skill columns are normally stored as string representations
    of lists, for example:
        "['python', 'java', 'sql']"
    """
    if isinstance(value, list):
        return value

    if isinstance(value, (tuple, set)):
        return list(value)

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)

            return []
        except (ValueError, SyntaxError):
            return []

    return []


def load_validation_data():
    """
    Load and validate the validation portion of training_pairs.csv.
    """
    if not TRAINING_PAIRS_PATH.is_file():
        raise FileNotFoundError(
            f"Training data was not found at: {TRAINING_PAIRS_PATH}"
        )

    df = pd.read_csv(TRAINING_PAIRS_PATH)

    required_columns = {
        "cv_text",
        "job_text",
        "cv_skills",
        "job_skills",
        "label",
        "split",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "training_pairs.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    validation_df = df[df["split"] == "val"].copy()

    if validation_df.empty:
        raise ValueError(
            "No rows with split='val' were found in training_pairs.csv."
        )

    return validation_df


def precompute_validation_components(
    validation_df,
    semantic_matcher,
):
    """
    Calculate the expensive NLP components once for every validation pair.

    Previously, the semantic model was executed again for every possible
    weight combination. That was unnecessary because changing the weights
    does not change the model outputs.

    All scores are normalised from [0, 100] to [0, 1] before optimisation.
    """

    component_rows = []
    total_rows = len(validation_df)

    for position, (_, row) in enumerate(
        validation_df.iterrows(),
        start=1,
    ):
        cv_text = str(row["cv_text"])
        job_text = str(row["job_text"])

        cv_skills = safe_list(row["cv_skills"])
        job_skills = safe_list(row["job_skills"])

        baseline = baseline_hybrid_score(
            cv_text,
            job_text,
            cv_skills,
            job_skills,
        )

        # This uses the exact same semantic pipeline as the application:
        # cleaning, chunking, cosine comparison and score conversion.
        semantic_score = semantic_matcher.similarity_score(
            cv_text,
            job_text,
        )

        component_rows.append({
            "label": float(row["label"]),

            # Convert all component scores to the [0, 1] interval.
            "semantic_score": semantic_score / 100.0,
            "baseline_score": baseline["baseline_score"] / 100.0,
            "skill_score": baseline["skill_score"] / 100.0,
        })

        if position % 100 == 0 or position == total_rows:
            print(
                f"Precomputed {position}/{total_rows} validation pairs"
            )

    return pd.DataFrame(component_rows)


def optimize_weights():
    """
    Select the best hybrid weights using the validation set.
    """
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            "The fine-tuned model is missing. Expected directory: "
            f"{MODEL_PATH}"
        )

    validation_df = load_validation_data()

    semantic_matcher = SemanticMatcher(
        str(MODEL_PATH)
    )

    components_df = precompute_validation_components(
        validation_df,
        semantic_matcher,
    )

    y_true = components_df["label"].to_numpy(
        dtype=float
    )

    semantic_scores = components_df[
        "semantic_score"
    ].to_numpy(dtype=float)

    baseline_scores = components_df[
        "baseline_score"
    ].to_numpy(dtype=float)

    skill_scores = components_df[
        "skill_score"
    ].to_numpy(dtype=float)

    results = []

    for semantic_weight in SEMANTIC_WEIGHTS:
        for baseline_weight in BASELINE_WEIGHTS:
            skill_weight = round(
                1.0
                - semantic_weight
                - baseline_weight,
                2,
            )

            # Skip combinations whose weights would exceed 1.
            if skill_weight < 0.0:
                continue

            predictions = (
                semantic_weight * semantic_scores
                + baseline_weight * baseline_scores
                + skill_weight * skill_scores
            )

            mae = mean_absolute_error(
                y_true,
                predictions,
            )

            rmse = (
                mean_squared_error(
                    y_true,
                    predictions,
                )
                ** 0.5
            )

            results.append({
                "semantic_weight": semantic_weight,
                "baseline_weight": baseline_weight,
                "skill_weight": skill_weight,
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4),
            })

    if not results:
        raise RuntimeError(
            "The configured weight grid produced no valid combinations."
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            ["RMSE", "MAE"],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    results_df.to_csv(
        OPTIMIZATION_REPORT_PATH,
        index=False,
    )

    best = results_df.iloc[0]

    print("\nTop validation combinations:")
    print(
        results_df
        .head(10)
        .to_string(index=False)
    )

    print("\nBest validation weights:")
    print(
        f"semantic_weight = "
        f"{best['semantic_weight']:.2f}"
    )
    print(
        f"baseline_weight = "
        f"{best['baseline_weight']:.2f}"
    )
    print(
        f"skill_weight = "
        f"{best['skill_weight']:.2f}"
    )
    print(
        f"MAE = {best['MAE']:.4f}"
    )
    print(
        f"RMSE = {best['RMSE']:.4f}"
    )

    print(
        f"\nSaved: {OPTIMIZATION_REPORT_PATH}"
    )

    print(
        "Copy the three best weights into "
        "src/final_matcher.py before running the evaluation."
    )


if __name__ == "__main__":
    optimize_weights()