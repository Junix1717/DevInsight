
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "error_classifier.joblib"

EVALUATION_PATH = (
    BASE_DIR / "datasets" / "evaluation" / "unseen_errors.csv"
)


def main():
    print("=== DEVINSIGHT - EVALUACIÓN ===")

    # Cargar modelo y dataset
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(EVALUATION_PATH)

    # Validar el dataset
    required_columns = {"error_message", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError("Faltan columnas en el dataset.")

    if df[["error_message", "label"]].isna().any().any():
        raise ValueError("El dataset contiene valores vacíos.")

    if not df["label"].isin(model.classes_).all():
        raise ValueError("El dataset contiene categorías desconocidas.")

    # Preparar los datos
    messages = df["error_message"]
    expected_labels = df["label"]

    # Predecir todos los mensajes
    predicted_labels = model.predict(messages)

    correct = (expected_labels == predicted_labels).sum()
    total = len(df)

    accuracy = correct / total * 100

    # Mostrar resultados individuales
    for message, expected, prediction in zip(
        messages, expected_labels, predicted_labels
    ):
        print(f"\nError: {message}")
        print(f"Esperado: {expected}")
        print(f"Predicción: {prediction}")

        result = (
            "CORRECTO"
            if expected == prediction
            else "INCORRECTO"
        )

        print(f"Resultado: {result}")

    # Resumen
    print("\n=== RESUMEN ===")
    print(f"Total de pruebas: {total}")
    print(f"Predicciones correctas: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Métricas por categoría
    print("\n=== REPORTE DE CLASIFICACIÓN ===")

    print(
        classification_report(
            expected_labels,
            predicted_labels,
            labels=model.classes_,
            zero_division=0
        )
    )

    # Matriz de confusión
    print("\n=== MATRIZ DE CONFUSIÓN ===")

    matrix = confusion_matrix(
        expected_labels,
        predicted_labels,
        labels=model.classes_
    )

    print(matrix)
    print("\nOrden de categorías:")

    for index, category in enumerate(model.classes_):
        print(f"  {index}: {category}")


if __name__ == "__main__":
    main()
