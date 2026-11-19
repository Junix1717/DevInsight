import joblib

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Ubicación de archivos
BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR / "datasets" / "processed" / "errors_expanded.csv"
)

# 1. Cargar el dataset
df = pd.read_csv(DATASET_PATH)

print("=== DEVINSIGHT - ML TRAINING ===")
print(f"Registros disponibles: {len(df)}")

# 2. Separar características y etiquetas
X = df["error_message"].fillna("")
y = df["label"]

# 3. Dividir entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nEntrenamiento: {len(X_train)} registros")
print(f"Prueba: {len(X_test)} registros")

# 4. Crear el pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# 5. Entrenar el modelo
print("\nEntrenando modelo...")

model.fit(X_train, y_train)

print("Entrenamiento completado.")

# 6. Evaluar el modelo
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n=== RESULTADOS ===")
print(f"Accuracy: {accuracy:.2%}")

print("\nReporte por categoría:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 7. Probar mensajes nuevos
examples = [
    "name 'cliente' is not defined",
    "'list' object has no attribute 'upper'",
    "invalid syntax",
    "division by zero"
]

print("\n=== PREDICCIONES ===")

for message, prediction in zip(
    examples,
    model.predict(examples)
):
    print(f"\nError: {message}")
    print(f"Categoría predicha: {prediction}")

# 8. Guardar el modelo entrenado
MODEL_PATH = BASE_DIR / "models" / "error_classifier.joblib"

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(model, MODEL_PATH)

print("\nModelo guardado correctamente.")
print(f"Ubicación: {MODEL_PATH}")