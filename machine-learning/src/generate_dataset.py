
import random
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "datasets" / "raw" / "errors.csv"
OUTPUT_PATH = BASE_DIR / "datasets" / "processed" / "errors_expanded.csv"

random.seed(42)

TARGET_PER_CLASS = 50

# Plantillas para generar mensajes
templates = {
    "NameError": [
        "name '{x}' is not defined",
        "name '{x}' is not defined. Did you mean: '{y}'?"
    ],
    "SyntaxError": [
        "invalid syntax near '{x}'",
        "expected ':' after '{x}'",
        "invalid syntax at line {n}"
    ],
    "TypeError": [
        "unsupported operand type(s) for +: '{x}' and '{y}'",
        "'{x}' object is not callable",
        "can only concatenate str (not '{x}') to str"
    ],
    "ImportError": [
        "No module named '{x}'",
        "cannot import name '{x}' from '{y}'"
    ],
    "AttributeError": [
        "'{x}' object has no attribute '{y}'",
        "type object '{x}' has no attribute '{y}'"
    ],
    "Other": [
        "division by zero",
        "list index out of range at position {n}",
        "KeyError: '{x}'",
        "ValueError: invalid literal '{x}'"
    ]
}

words = [
    "usuario", "resultado", "contador", "precio",
    "total", "nombre", "edad", "producto",
    "cantidad", "promedio", "datos", "archivo",
    "config", "cliente", "pedido", "lista",
    "valor", "indice", "numero", "mensaje"
]

def generate_examples(label, count):
    examples = []
    seen = set()
    attempts = 0

    while len(examples) < count and attempts < 10000:
        attempts += 1

        template = random.choice(templates[label])

        x = random.choice(words)
        y = random.choice(words)
        n = random.randint(1, 100)

        message = template.format(x=x, y=y, n=n)

        if message in seen:
            continue

        seen.add(message)

        examples.append({
            "error_message": message,
            "code_snippet": "",
            "label": label
        })

    return examples


def main():
    original_df = pd.read_csv(RAW_PATH)

    generated = []

    for label in templates:
        generated.extend(
            generate_examples(label, TARGET_PER_CLASS)
        )

    generated_df = pd.DataFrame(generated)

    combined = pd.concat(
        [original_df, generated_df],
        ignore_index=True
    )

    # Evitar repetir el mismo mensaje y categoría
    combined = combined.drop_duplicates(
        subset=["error_message", "label"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(OUTPUT_PATH, index=False)

    print("Dataset generado correctamente.")
    print(f"Total de registros: {len(combined)}")

    print("\nDistribución por categoría:")

    for category, count in combined["label"].value_counts().items():
        print(f"  {category:<20} {count:>3}")


if __name__ == "__main__":
    main()
