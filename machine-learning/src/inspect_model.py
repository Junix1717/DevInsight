
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "error_classifier.joblib"

# Cargar nuestro modelo entrenado
model = joblib.load(MODEL_PATH)

# Obtener los componentes del Pipeline
vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

# Obtener las palabras y expresiones aprendidas
features = vectorizer.get_feature_names_out()

# Obtener las categorías
categories = classifier.classes_

print("=== INSPECCIÓN DEL MODELO ===")
print(f"Características aprendidas: {len(features)}")
print(f"Categorías: {len(categories)}")

# Mostrar las 10 características con mayor peso
# positivo para cada categoría
for index, category in enumerate(categories):

    weights = classifier.coef_[index]

    top_indices = weights.argsort()[-10:][::-1]

    print(f"\n=== {category} ===")

    for feature_index in top_indices:
        feature = features[feature_index]
        weight = weights[feature_index]

        print(f"{feature:<25} {weight:+.4f}")
