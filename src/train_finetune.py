import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)

from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TRAINING_PATH = (
    DATA_DIR
    / "training_pairs.csv"
)

OUTPUT_DIR = (
    MODELS_DIR
    / "fine_tuned_sbert"
)

METADATA_PATH = (
    REPORTS_DIR
    / "training_metadata.json"
)

BASE_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

RANDOM_SEED = 42
BATCH_SIZE = 8
EPOCHS = 1
MAX_SEQUENCE_LENGTH = 256


def set_reproducible_seeds(
    seed: int = RANDOM_SEED,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def load_training_rows() -> pd.DataFrame:
    if not TRAINING_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {TRAINING_PATH}. "
            "Run: python -m src.create_training_pairs"
        )

    df = pd.read_csv(
        TRAINING_PATH
    )

    required_columns = {
        "cv_text",
        "job_text",
        "label",
        "split",
    }

    missing_columns = (
        required_columns
        .difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "training_pairs.csv is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    train_df = df[
        df["split"] == "train"
    ].copy()

    if train_df.empty:
        raise ValueError(
            "No rows with split='train' were found."
        )

    return train_df


def train_finetuned_model() -> None:
    set_reproducible_seeds()

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = load_training_rows()

    model = SentenceTransformer(
        BASE_MODEL
    )

    model.max_seq_length = (
        MAX_SEQUENCE_LENGTH
    )

    train_examples = [
        InputExample(
            texts=[
                str(row["cv_text"]),
                str(row["job_text"]),
            ],
            label=float(row["label"]),
        )
        for _, row in train_df.iterrows()
    ]

    generator = torch.Generator()
    generator.manual_seed(
        RANDOM_SEED
    )

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE,
        generator=generator,
    )

    train_loss = (
        losses.CosineSimilarityLoss(
            model=model
        )
    )

    total_training_steps = (
        len(train_dataloader)
        * EPOCHS
    )

    warmup_steps = max(
        1,
        round(
            total_training_steps
            * 0.10
        ),
    )

    if OUTPUT_DIR.exists():
        shutil.rmtree(
            OUTPUT_DIR
        )

    model.fit(
        train_objectives=[
            (
                train_dataloader,
                train_loss,
            )
        ],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=str(
            OUTPUT_DIR
        ),
        show_progress_bar=True,
    )

    metadata = {
        "base_model": BASE_MODEL,
        "random_seed": RANDOM_SEED,
        "training_pairs": len(train_df),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "warmup_steps": warmup_steps,
        "loss": "CosineSimilarityLoss",
        "split_strategy": (
            "candidate-grouped 70/15/15"
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"Fine-tuned model saved to: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Training metadata saved to: "
        f"{METADATA_PATH}"
    )


if __name__ == "__main__":
    train_finetuned_model()