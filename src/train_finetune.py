from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer import losses
from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def train_finetuned_model():
    MODELS_DIR.mkdir(exist_ok=True)

    training_path = DATA_DIR / "training_pairs.csv"
    output_dir = MODELS_DIR / "fine_tuned_sbert"

    df = pd.read_csv(training_path)
    train_df = df[df["split"] == "train"].copy()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    train_examples = [
        InputExample(
            texts=[str(row["cv_text"]), str(row["job_text"])],
            label=float(row["label"])
        )
        for _, row in train_df.iterrows()
    ]

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=8
    )

    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=10,
        output_path=str(output_dir),
        show_progress_bar=True
    )

    print(f"Model salvat în: {output_dir}")


if __name__ == "__main__":
    train_finetuned_model()