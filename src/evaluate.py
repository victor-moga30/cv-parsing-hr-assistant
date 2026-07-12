import ast
import math
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from src.baseline import baseline_hybrid_score
from src.final_matcher import FinalMatcher
from src.semantic_matcher import SemanticMatcher


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

TRAINING_PAIRS_PATH = DATA_DIR / "training_pairs.csv"
FINETUNED_MODEL_PATH = MODELS_DIR / "fine_tuned_sbert"

OPTIMIZATION_REPORT_PATH = (
    REPORTS_DIR / "optimization_results.csv"
)

METRICS_PATH = (
    REPORTS_DIR / "model_metrics.csv"
)

PREDICTIONS_PATH = (
    REPORTS_DIR / "evaluation_predictions.csv"
)

CLASSIFICATION_THRESHOLD = 0.50


def safe_list(value):
    """
    Convert a value loaded from the CSV into a Python list.
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


def load_test_data():
    """
    Load and validate the test portion of training_pairs.csv.
    """
    if not TRAINING_PAIRS_PATH.is_file():
        raise FileNotFoundError(
            f"Training data was not found at: "
            f"{TRAINING_PAIRS_PATH}"
        )

    df = pd.read_csv(
        TRAINING_PAIRS_PATH
    )

    required_columns = {
        "cv_text",
        "job_text",
        "cv_skills",
        "job_skills",
        "label",
        "split",
    }

    missing_columns = required_columns.difference(
        df.columns
    )

    if missing_columns:
        raise ValueError(
            "training_pairs.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    test_df = df[
        df["split"] == "test"
    ].copy()

    if test_df.empty:
        raise ValueError(
            "No rows with split='test' were found "
            "in training_pairs.csv."
        )

    return test_df


def verify_runtime_weights(final_matcher):
    """
    Verify that FinalMatcher uses the best weights saved by optimisation.

    This prevents evaluating one set of weights while the Streamlit
    application uses a different set.
    """
    if not OPTIMIZATION_REPORT_PATH.is_file():
        raise FileNotFoundError(
            "Optimization report was not found.\n"
            "Run this command first:\n"
            "python -m src.optimize_matcher"
        )

    optimization_df = pd.read_csv(
        OPTIMIZATION_REPORT_PATH
    )

    required_columns = {
        "semantic_weight",
        "baseline_weight",
        "skill_weight",
        "RMSE",
    }

    missing_columns = required_columns.difference(
        optimization_df.columns
    )

    if missing_columns:
        raise ValueError(
            "optimization_results.csv is missing "
            "required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if optimization_df.empty:
        raise ValueError(
            "optimization_results.csv is empty."
        )

    sort_columns = ["RMSE"]

    if "MAE" in optimization_df.columns:
        sort_columns.append("MAE")

    best = (
        optimization_df
        .sort_values(
            sort_columns,
            ascending=True,
        )
        .iloc[0]
    )

    optimized_weights = {
        "semantic_weight": float(
            best["semantic_weight"]
        ),
        "baseline_weight": float(
            best["baseline_weight"]
        ),
        "skill_weight": float(
            best["skill_weight"]
        ),
    }

    runtime_weights = {
        "semantic_weight": float(
            final_matcher.semantic_weight
        ),
        "baseline_weight": float(
            final_matcher.baseline_weight
        ),
        "skill_weight": float(
            final_matcher.skill_weight
        ),
    }

    mismatches = []

    for weight_name in optimized_weights:
        optimized_value = optimized_weights[
            weight_name
        ]

        runtime_value = runtime_weights[
            weight_name
        ]

        if not math.isclose(
            optimized_value,
            runtime_value,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            mismatches.append(
                f"{weight_name}: "
                f"optimized={optimized_value:.2f}, "
                f"runtime={runtime_value:.2f}"
            )

    if mismatches:
        raise ValueError(
            "FinalMatcher does not use the best "
            "optimization weights.\n"
            + "\n".join(mismatches)
            + "\nUpdate the three weights in "
            "src/final_matcher.py and rerun evaluation."
        )

    return runtime_weights


def safe_corr(
    correlation_function,
    y_true,
    predictions,
):
    """
    Calculate correlation safely.

    Pearson or Spearman correlation may be undefined when one
    of the arrays is constant.
    """
    try:
        value = correlation_function(
            y_true,
            predictions,
        )[0]

        value = float(value)

        if not math.isfinite(value):
            return 0.0

        return round(value, 4)

    except Exception:
        return 0.0


def to_binary(
    values,
    threshold,
):
    """
    Convert continuous matching scores into binary classes.
    """
    return [
        1 if float(value) >= threshold else 0
        for value in values
    ]


def add_metrics(
    rows,
    model_name,
    y_true,
    predictions,
    threshold,
):
    """
    Calculate regression and classification metrics.
    """
    mse = mean_squared_error(
        y_true,
        predictions,
    )

    y_true_class = to_binary(
        y_true,
        threshold,
    )

    y_pred_class = to_binary(
        predictions,
        threshold,
    )

    rows.append({
        "model": model_name,

        "MAE": round(
            float(
                mean_absolute_error(
                    y_true,
                    predictions,
                )
            ),
            4,
        ),

        "MSE": round(
            float(mse),
            4,
        ),

        "RMSE": round(
            float(mse ** 0.5),
            4,
        ),

        "Pearson": safe_corr(
            pearsonr,
            y_true,
            predictions,
        ),

        "Spearman": safe_corr(
            spearmanr,
            y_true,
            predictions,
        ),

        "Accuracy": round(
            float(
                accuracy_score(
                    y_true_class,
                    y_pred_class,
                )
            ),
            4,
        ),

        "Precision": round(
            float(
                precision_score(
                    y_true_class,
                    y_pred_class,
                    zero_division=0,
                )
            ),
            4,
        ),

        "Recall": round(
            float(
                recall_score(
                    y_true_class,
                    y_pred_class,
                    zero_division=0,
                )
            ),
            4,
        ),

        "F1_score": round(
            float(
                f1_score(
                    y_true_class,
                    y_pred_class,
                    zero_division=0,
                )
            ),
            4,
        ),

        "Threshold": threshold,
    })


def save_confusion_and_report(
    model_name,
    y_true,
    predictions,
    threshold,
):
    """
    Save the confusion matrix and classification report for one model.
    """
    y_true_class = to_binary(
        y_true,
        threshold,
    )

    y_pred_class = to_binary(
        predictions,
        threshold,
    )

    matrix = confusion_matrix(
        y_true_class,
        y_pred_class,
        labels=[0, 1],
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "actual_not_match",
            "actual_match",
        ],
        columns=[
            "predicted_not_match",
            "predicted_match",
        ],
    )

    matrix_df.to_csv(
        REPORTS_DIR
        / f"confusion_matrix_{model_name}.csv"
    )

    report = classification_report(
        y_true_class,
        y_pred_class,
        labels=[0, 1],
        target_names=[
            "not_match",
            "match",
        ],
        zero_division=0,
    )

    report_path = (
        REPORTS_DIR
        / f"classification_report_{model_name}.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)


def evaluate():
    """
    Evaluate:

    1. Classical TF-IDF + skill baseline
    2. Pretrained SBERT using the runtime semantic pipeline
    3. Fine-tuned SBERT using the runtime semantic pipeline
    4. The complete final hybrid pipeline used by the application
    """
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not FINETUNED_MODEL_PATH.is_dir():
        raise FileNotFoundError(
            "The fine-tuned model is missing. "
            "Expected directory: "
            f"{FINETUNED_MODEL_PATH}"
        )

    test_df = load_test_data()

    y_true = (
        test_df["label"]
        .astype(float)
        .tolist()
    )

    # Pretrained model evaluated with the same semantic pipeline
    # as the application: chunking, cosine comparison and score
    # conversion.
    pretrained_matcher = SemanticMatcher(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # FinalMatcher loads:
    # - the fine-tuned model;
    # - the weights used by the live application.
    final_matcher = FinalMatcher(
        model_name=str(
            FINETUNED_MODEL_PATH
        )
    )

    runtime_weights = verify_runtime_weights(
        final_matcher
    )

    # Reuse the model already loaded by FinalMatcher.
    # This avoids loading the fine-tuned model twice.
    finetuned_matcher = (
        final_matcher.semantic_matcher
    )

    baseline_predictions = []
    pretrained_predictions = []
    finetuned_predictions = []
    final_hybrid_predictions = []

    prediction_rows = []

    total_rows = len(test_df)

    for position, (
        row_index,
        row,
    ) in enumerate(
        test_df.iterrows(),
        start=1,
    ):
        cv_text = str(
            row["cv_text"]
        )

        job_text = str(
            row["job_text"]
        )

        cv_skills = safe_list(
            row["cv_skills"]
        )

        job_skills = safe_list(
            row["job_skills"]
        )

        baseline = baseline_hybrid_score(
            cv_text,
            job_text,
            cv_skills,
            job_skills,
        )

        baseline_score = (
            baseline["baseline_score"]
            / 100.0
        )

        skill_score = (
            baseline["skill_score"]
            / 100.0
        )

        pretrained_score = (
            pretrained_matcher.similarity_score(
                cv_text,
                job_text,
            )
            / 100.0
        )

        finetuned_score = (
            finetuned_matcher.similarity_score(
                cv_text,
                job_text,
            )
            / 100.0
        )

        final_hybrid_score = (
            runtime_weights["semantic_weight"]
            * finetuned_score

            + runtime_weights["baseline_weight"]
            * baseline_score

            + runtime_weights["skill_weight"]
            * skill_score
        )

        final_hybrid_score = max(
            0.0,
            min(
                1.0,
                final_hybrid_score,
            ),
        )

        baseline_predictions.append(
            baseline_score
        )

        pretrained_predictions.append(
            pretrained_score
        )

        finetuned_predictions.append(
            finetuned_score
        )

        final_hybrid_predictions.append(
            final_hybrid_score
        )

        prediction_row = {
            "row_index": row_index,
            "true_label": float(
                row["label"]
            ),

            "baseline_prediction": round(
                baseline_score,
                6,
            ),

            "pretrained_sbert_prediction": round(
                pretrained_score,
                6,
            ),

            "fine_tuned_sbert_prediction": round(
                finetuned_score,
                6,
            ),

            "final_hybrid_prediction": round(
                final_hybrid_score,
                6,
            ),
        }

        # Include IDs for debugging when they exist,
        # without saving CV text or job-description text.
        for optional_column in (
            "candidate_id",
            "job_id",
        ):
            if optional_column in test_df.columns:
                prediction_row[
                    optional_column
                ] = row[optional_column]

        prediction_rows.append(
            prediction_row
        )

        if (
            position % 100 == 0
            or position == total_rows
        ):
            print(
                f"Evaluated "
                f"{position}/{total_rows} test pairs"
            )

    metric_inputs = [
        (
            "baseline_tfidf_skill",
            baseline_predictions,
        ),
        (
            "pretrained_sbert_runtime_pipeline",
            pretrained_predictions,
        ),
        (
            "fine_tuned_sbert_runtime_pipeline",
            finetuned_predictions,
        ),
        (
            "final_hybrid_runtime_pipeline",
            final_hybrid_predictions,
        ),
    ]

    metric_rows = []

    for (
        model_name,
        predictions,
    ) in metric_inputs:
        add_metrics(
            metric_rows,
            model_name,
            y_true,
            predictions,
            CLASSIFICATION_THRESHOLD,
        )

        save_confusion_and_report(
            model_name,
            y_true,
            predictions,
            CLASSIFICATION_THRESHOLD,
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    pd.DataFrame(
        prediction_rows
    ).to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    positive_count = sum(
        to_binary(
            y_true,
            CLASSIFICATION_THRESHOLD,
        )
    )

    negative_count = (
        len(y_true)
        - positive_count
    )

    print("\nRuntime weights verified:")

    print(
        f"semantic_weight = "
        f"{runtime_weights['semantic_weight']:.2f}"
    )

    print(
        f"baseline_weight = "
        f"{runtime_weights['baseline_weight']:.2f}"
    )

    print(
        f"skill_weight = "
        f"{runtime_weights['skill_weight']:.2f}"
    )

    print(
        f"\nTest distribution at threshold "
        f"{CLASSIFICATION_THRESHOLD:.2f}:"
    )

    print(
        f"not_match = {negative_count}"
    )

    print(
        f"match = {positive_count}"
    )

    print("\nEvaluation metrics:")

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {METRICS_PATH}"
    )

    print(
        f"Saved: {PREDICTIONS_PATH}"
    )

    print(
        "Saved: confusion_matrix_*.csv"
    )

    print(
        "Saved: classification_report_*.txt"
    )


if __name__ == "__main__":
    evaluate()