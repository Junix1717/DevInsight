from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "error_classifier.joblib"

# Casos de evaluación
test_cases = [
    (
        "RecursionError: maximum recursion depth exceeded",
        "Other"
    ),
    (
        "FileNotFoundError: [Errno 2] No such file or directory: 'datos.csv'",
        "Other"
    ),
    (
        "IndentationError: unexpected indent",
        "SyntaxError"
    ),
    (
        "NameError: name 'calcular_total' is not defined",
        "NameError"
    ),
    (
        "AttributeError: 'NoneType' object has no attribute 'split'",
        "AttributeError"
    )
]


def main():
    model = joblib.load(MODEL_PATH)

    correct = 0

    print("=== DEVINSIGHT - EVALUACIÓN ===")

    expected_labels = []
    predicted_labels = []

    for message, expected in test_cases:
        prediction = model.predict([message])[0]

        expected_labels.append(expected)
        predicted_labels.append(prediction)

        is_correct = prediction == expected
        

        if is_correct:
            correct += 1

        print(f"\nError: {message}")
        print(f"Esperado: {expected}")
        print(f"Predicción: {prediction}")
        print(f"Resultado: {'CORRECTO' if is_correct else 'INCORRECTO'}")

    total = len(test_cases)
    accuracy = correct / total * 100

    print("\n=== RESUMEN ===")
    print(f"Total de pruebas: {total}")
    print(f"Predicciones correctas: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    print("\n=== REPORTE DE CLASIFICACIÓN ===")

    print(
        classification_report(
            expected_labels,
            predicted_labels,
            zero_division=0
        )
    )

    print("\n=== MATRIZ DE CONFUSIÓN ===")

    print(
        confusion_matrix(
            expected_labels,
            predicted_labels,
            labels=model.classes_
        )
    )

    print("Orden de categorías:", model.classes_)


if __name__ == "__main__":
    main()
