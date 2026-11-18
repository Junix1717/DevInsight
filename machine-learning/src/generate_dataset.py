
import pandas as pd
from pathlib import Path

# Ubicación del dataset original
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "datasets" / "raw" / "errors.csv"

# Leer los datos existentes
df = pd.read_csv(DATASET_PATH)

new_data = []

# Ejemplos de nombres de variables
variables = [
    "usuario",
    "resultado",
    "contador",
    "precio",
    "total",
    "nombre",
    "edad",
    "producto",
    "cantidad",
    "promedio"
]

# Generar ejemplos NameError
for variable in variables:
    new_data.append({
        "error_message": f"name '{variable}' is not defined",
        "code_snippet": f"print({variable})",
        "label": "NameError"
    })

# Generar ejemplos AttributeError
attributes = [
    ("str", "append"),
    ("int", "upper"),
    ("list", "split"),
    ("tuple", "add"),
    ("float", "lower"),
    ("dict", "push"),
    ("set", "sort"),
    ("bool", "strip"),
    ("list", "encode"),
    ("str", "pop")
]

for data_type, attribute in attributes:
    new_data.append({
        "error_message": (
            f"'{data_type}' object has no attribute '{attribute}'"
        ),
        "code_snippet": f"obj.{attribute}()",
        "label": "AttributeError"
    })

# Convertir los ejemplos a DataFrame
new_df = pd.DataFrame(new_data)

# Combinar con los registros originales
combined_df = pd.concat(
    [df, new_df],
    ignore_index=True
)

# Eliminar duplicados
combined_df = combined_df.drop_duplicates()

# Guardar en processed para conservar el original
OUTPUT_PATH = (
    BASE_DIR / "datasets" / "processed" / "errors_expanded.csv"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

combined_df.to_csv(OUTPUT_PATH, index=False)

print("Dataset ampliado correctamente.")
print(f"Registros originales: {len(df)}")
print(f"Registros totales: {len(combined_df)}")

print("\nDistribución de categorías:")
print(combined_df["label"].value_counts())
