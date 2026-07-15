"""
Lectura y validacion del dataset de gestion ganadera.

Expone utilidades para cargar el CSV, verificar su integridad y generar un
resumen textual que se inyecta en el system prompt del agente.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columnas de fecha que se parsean a datetime
DATE_COLUMNS = ["fecha_nacimiento", "fecha_destete", "fecha_evento"]

# Columnas categoricas del dominio ganadero
CATEGORICAL_COLUMNS = ["finca", "raza", "sexo", "estado"]

# Conjunto completo de columnas esperadas en el dataset
EXPECTED_COLUMNS = [
    "id_animal",
    "codani",
    "finca",
    "raza",
    "sexo",
    "fecha_nacimiento",
    "peso_nacimiento",
    "fecha_destete",
    "peso_destete",
    "id_madre",
    "id_toro",
    "estado",
    "fecha_evento",
    "costo_alimentacion_mensual",
]


def load_dataset(path: str) -> pd.DataFrame:
    """Carga el dataset ganadero desde un archivo CSV.

    Parsea las columnas de fecha a datetime y convierte las columnas
    categoricas al tipo `category` para un uso mas eficiente de memoria.

    Args:
        path: Ruta al archivo CSV.

    Returns:
        DataFrame con los tipos ya normalizados.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
    """
    # Verificar existencia antes de intentar leer
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Error: No se encontró el archivo de datos en {path}"
        )

    # Leer el CSV parseando las fechas directamente
    df = pd.read_csv(path, parse_dates=DATE_COLUMNS)

    # Convertir las columnas categoricas presentes al tipo category
    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("category")

    # codani es un identificador, no un numero: mantenerlo como texto
    if "codani" in df.columns:
        df["codani"] = df["codani"].astype(str)

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Valida la integridad estructural del dataset ganadero.

    Comprueba que esten todas las columnas esperadas y que el identificador
    `codani` sea unico.

    Args:
        df: DataFrame a validar.

    Raises:
        ValueError: Si faltan columnas o si `codani` contiene duplicados.
    """
    # Verificar que no falte ninguna columna esperada
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Error: Faltan columnas esperadas en el dataset: "
            + ", ".join(missing)
        )

    # Verificar unicidad del identificador SENASA
    if df["codani"].duplicated().any():
        duplicados = int(df["codani"].duplicated().sum())
        raise ValueError(
            f"Error: La columna 'codani' debe ser única, se encontraron "
            f"{duplicados} valores duplicados"
        )


def get_dataset_summary(df: pd.DataFrame) -> str:
    """Genera un resumen textual del dataset para el system prompt.

    Incluye cantidad de filas, columnas con sus tipos, rangos de las columnas
    numericas y de fecha, y los valores unicos de las categoricas. El objetivo
    es dar al agente contexto suficiente sobre la estructura de los datos.

    Args:
        df: DataFrame ya cargado y validado.

    Returns:
        Cadena de texto con el resumen formateado.
    """
    lines: list[str] = []
    lines.append(f"El dataset contiene {len(df)} registros y "
                 f"{len(df.columns)} columnas.")

    # Detalle de columnas y sus tipos
    lines.append("\nColumnas y tipos:")
    for column in df.columns:
        lines.append(f"  - {column}: {df[column].dtype}")

    # Rangos de las columnas de fecha
    date_cols = [c for c in DATE_COLUMNS if c in df.columns]
    if date_cols:
        lines.append("\nRangos de fechas:")
        for column in date_cols:
            valores = df[column].dropna()
            if valores.empty:
                lines.append(f"  - {column}: sin datos")
            else:
                fmin = valores.min().date()
                fmax = valores.max().date()
                lines.append(f"  - {column}: {fmin} a {fmax}")

    # Rangos de las columnas numericas
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        lines.append("\nRangos numericos (min / promedio / max):")
        for column in numeric_cols:
            serie = df[column].dropna()
            if serie.empty:
                lines.append(f"  - {column}: sin datos")
            else:
                lines.append(
                    f"  - {column}: {serie.min():.1f} / "
                    f"{serie.mean():.1f} / {serie.max():.1f}"
                )

    # Valores unicos de las categoricas
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    if cat_cols:
        lines.append("\nValores de las columnas categoricas:")
        for column in cat_cols:
            valores = sorted(str(v) for v in df[column].dropna().unique())
            lines.append(f"  - {column}: {', '.join(valores)}")

    # Porcentaje de nulos por columna con datos faltantes
    nulos = df.isna().mean().mul(100).round(1)
    con_nulos = nulos[nulos > 0]
    if not con_nulos.empty:
        lines.append("\nColumnas con valores nulos (porcentaje):")
        for column, pct in con_nulos.items():
            lines.append(f"  - {column}: {pct}%")

    return "\n".join(lines)
