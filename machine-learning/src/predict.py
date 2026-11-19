
from pathlib import Path
import joblib

# Ubicación del modelo entrenado
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "error_classifier.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el modelo entrenado. "
            "Ejecuta train.py primero."
        )

    return joblib.load(MODEL_PATH)


def predict_error(model, message):
    # Obtener la categoría
    prediction = model.predict([message])[0]

    # Obtener las probabilidades
    probabilities = model.predict_proba([message])[0]

    confidence = max(probabilities) * 100

    return prediction, confidence


def main():
    print("=== DEVINSIGHT - ERROR CLASSIFIER ===")

    # Cargar el modelo una sola vez
    model = load_model()

    while True:
        message = input(
            "\nIntroduce un error de Python (o 'salir'): "
        )

        if message.strip().lower() == "salir":
            print("\nFinalizando DevInsight...")
            break

        if not message.strip():
            print("Debes introducir un mensaje.")
            continue

        category, confidence = predict_error(model, message)

        print("\n=== RESULTADO ===")
        print(f"Categoría: {category}")
        print(f"Confianza estimada: {confidence:.2f}%")


if __name__ == "__main__":
    main()
  