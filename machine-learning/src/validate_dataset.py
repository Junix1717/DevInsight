
import pandas as pd
from pathlib import Path

DATASET_PATH = Path("machine-learning/datasets/raw/errors.csv")

REQUIRED_COLUMNS = [
    "error_message",
    "code_snippet",
    "label"
]

VALID_LABELS = {
    "SyntaxError",
    "NameError",
    "TypeError",
    "ImportError",
    "AttributeError",
    "Other"
}


def validate_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    # Comprobar columnas
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas: {missing}"
        )

    # Comprobar valores vacíos
    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(
            "El dataset contiene valores vacíos."
        )

    # Comprobar categorías
    invalid_labels = set(df["label"]) - VALID_LABELS

    if invalid_labels:
        raise ValueError(
            f"Categorías desconocidas: {invalid_labels}"
        )

    print("\nDataset válido")
    print(f"Total de registros: {len(df)}")

    print("\nDistribución de categorías:")
    print(df["label"].value_counts())

    print("\nDuplicados:")
    print(df.duplicated().sum())


if __name__ == "__main__":
    validate_dataset()
