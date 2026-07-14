"""
Generador de dataset sintetico de gestion ganadera para el proyecto Alura Agente.

Produce data/ganaderia_dataset.csv de forma reproducible (semilla fija 42) con
datos coherentes de partos, pesos, fincas y razas. El objetivo es alimentar al
agente LangChain + Gemini con informacion realista para preguntas analiticas.

Uso:
    python scripts/generate_dataset.py
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Semilla fija para reproducibilidad
SEED = 42

# Rutas relativas a la raiz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "ganaderia_dataset.csv"

# Dominios categoricos del negocio
FINCAS = ["ECOS", "Nispero", "Las Flores", "La Libertad", "Miravalles"]
RAZAS = ["Brahman", "Angus", "Nelore", "Simmental", "Cruzado", "Jersey"]
ESTADOS = ["Activo", "Vendido", "Muerto", "Traslado"]

# Ventana temporal de nacimientos
FECHA_INICIO = date(2022, 1, 1)
FECHA_FIN = date(2025, 12, 31)

# Horizonte maximo para eventos posteriores (destete, cambio de estado)
HORIZONTE_MAX = date(2026, 6, 30)

# Pesos por anio para que 2024 tenga mas nacimientos que el resto
PROB_ANIO = {2022: 0.22, 2023: 0.23, 2024: 0.35, 2025: 0.20}

# Ajuste de peso al destete por finca (Miravalles consistentemente mas alto)
BONUS_DESTETE_FINCA = {
    "ECOS": 0.0,
    "Nispero": -5.0,
    "Las Flores": 0.0,
    "La Libertad": -3.0,
    "Miravalles": 28.0,
}


def _random_date_in_year(rng: np.random.Generator, year: int) -> date:
    # Elegir un dia aleatorio dentro del anio, respetando el rango global
    inicio = max(date(year, 1, 1), FECHA_INICIO)
    fin = min(date(year, 12, 31), FECHA_FIN)
    span = (fin - inicio).days
    offset = int(rng.integers(0, span + 1))
    return inicio + timedelta(days=offset)


def _clip_date(d: date, tope: date) -> date:
    # Evitar que un evento supere el horizonte de recoleccion de datos
    return min(d, tope)


def generate_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    # Cantidad de filas reproducible dentro del rango pedido
    n_rows = int(rng.integers(800, 1201))

    # Identificadores incrementales
    id_animales = [f"CR-{i:05d}" for i in range(1, n_rows + 1)]

    # codani: 10 digitos numericos unicos (identificador SENASA)
    codani_set: set[int] = set()
    while len(codani_set) < n_rows:
        faltantes = n_rows - len(codani_set)
        candidatos = rng.integers(1_000_000_000, 10_000_000_000, size=faltantes)
        codani_set.update(int(c) for c in candidatos)
    codani = [str(c) for c in list(codani_set)[:n_rows]]

    # Categoricas base
    fincas = rng.choice(FINCAS, size=n_rows)
    razas = rng.choice(RAZAS, size=n_rows)
    sexos = rng.choice(["M", "H"], size=n_rows)

    # Fechas de nacimiento con 2024 sobre-representado
    anios = list(PROB_ANIO.keys())
    probs = np.array(list(PROB_ANIO.values()))
    probs = probs / probs.sum()
    anios_elegidos = rng.choice(anios, size=n_rows, p=probs)
    fechas_nacimiento = [
        _random_date_in_year(rng, int(y)) for y in anios_elegidos
    ]

    # Peso al nacimiento: normal(32, 4) recortado a [22, 45]
    peso_nacimiento = np.clip(rng.normal(32.0, 4.0, size=n_rows), 22.0, 45.0)
    peso_nacimiento = np.round(peso_nacimiento, 1)

    # Toro asignado
    id_toro = [f"TORO-{int(rng.integers(1, 13)):02d}" for _ in range(n_rows)]

    # Costo de alimentacion mensual en colones CRC
    costo_alimentacion = rng.uniform(15000, 45000, size=n_rows)
    costo_alimentacion = np.round(costo_alimentacion, 0)

    # Estado del animal (mayoria activos)
    estados = rng.choice(ESTADOS, size=n_rows, p=[0.62, 0.20, 0.10, 0.08])

    # Construir columnas dependientes fila por fila para respetar coherencia
    fechas_destete: list[date | None] = []
    pesos_destete: list[float | None] = []
    fechas_evento: list[date | None] = []

    for i in range(n_rows):
        nacimiento = fechas_nacimiento[i]
        finca = fincas[i]
        estado = estados[i]

        # Fecha de evento: nula si Activo, posterior al nacimiento en otro caso
        fecha_evento: date | None = None
        if estado != "Activo":
            # Entre 10 y 900 dias despues del nacimiento
            dias_evento = int(rng.integers(10, 901))
            fecha_evento = _clip_date(
                nacimiento + timedelta(days=dias_evento), HORIZONTE_MAX
            )

        # Destete: 7-9 meses tras el nacimiento, nulo en ~15% de casos
        tiene_destete = rng.random() > 0.15
        fecha_destete: date | None = None
        peso_destete: float | None = None

        if tiene_destete:
            dias_destete = int(rng.integers(210, 271))  # aprox 7-9 meses
            candidata_destete = nacimiento + timedelta(days=dias_destete)

            # Un animal muerto antes del destete no llega a destetarse
            murio_antes = (
                estado == "Muerto"
                and fecha_evento is not None
                and fecha_evento < candidata_destete
            )
            # Tampoco tiene sentido un destete mas alla del horizonte de datos
            fuera_de_horizonte = candidata_destete > HORIZONTE_MAX

            if not murio_antes and not fuera_de_horizonte:
                fecha_destete = candidata_destete
                # Peso al destete: normal(180, 30) mas bonus por finca
                base = rng.normal(180.0, 30.0) + BONUS_DESTETE_FINCA[finca]
                peso = float(np.clip(base, 120.0, 260.0))
                # Garantizar que siempre supere el peso al nacimiento
                peso = max(peso, float(peso_nacimiento[i]) + 60.0)
                peso_destete = round(min(peso, 260.0), 1)

        fechas_destete.append(fecha_destete)
        pesos_destete.append(peso_destete)
        fechas_evento.append(fecha_evento)

    # id_madre: id_animal de una hembra nacida antes, nulo en ~20% de casos
    hembras_idx = [
        i for i in range(n_rows) if sexos[i] == "H"
    ]
    id_madre: list[str | None] = []
    for i in range(n_rows):
        if rng.random() < 0.20 or not hembras_idx:
            id_madre.append(None)
            continue
        # Candidatas: hembras nacidas antes que el animal actual
        candidatas = [
            j for j in hembras_idx
            if fechas_nacimiento[j] < fechas_nacimiento[i] and j != i
        ]
        if not candidatas:
            id_madre.append(None)
        else:
            elegida = int(rng.choice(candidatas))
            id_madre.append(id_animales[elegida])

    df = pd.DataFrame(
        {
            "id_animal": id_animales,
            "codani": codani,
            "finca": fincas,
            "raza": razas,
            "sexo": sexos,
            "fecha_nacimiento": fechas_nacimiento,
            "peso_nacimiento": peso_nacimiento,
            "fecha_destete": fechas_destete,
            "peso_destete": pesos_destete,
            "id_madre": id_madre,
            "id_toro": id_toro,
            "estado": estados,
            "fecha_evento": fechas_evento,
            "costo_alimentacion_mensual": costo_alimentacion,
        }
    )

    return df


def _validar_coherencia(df: pd.DataFrame) -> None:
    # Verificaciones internas de las reglas de negocio
    activos = df["estado"] == "Activo"
    assert df.loc[activos, "fecha_evento"].isna().all(), (
        "Un animal Activo no debe tener fecha_evento"
    )

    no_activos = ~activos
    con_evento = df.loc[no_activos & df["fecha_evento"].notna()]
    assert (
        con_evento["fecha_evento"] > con_evento["fecha_nacimiento"]
    ).all(), "fecha_evento debe ser posterior al nacimiento"

    # Destete y peso de destete deben ser nulos o presentes en conjunto
    assert (
        df["fecha_destete"].isna() == df["peso_destete"].isna()
    ).all(), "fecha_destete y peso_destete deben coincidir en nulidad"

    con_destete = df[df["peso_destete"].notna()]
    assert (
        con_destete["peso_destete"] > con_destete["peso_nacimiento"]
    ).all(), "peso_destete debe superar al peso_nacimiento"


def imprimir_resumen(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("RESUMEN DEL DATASET GENERADO")
    print("=" * 60)
    print(f"Cantidad de filas: {len(df)}")

    fmin = pd.to_datetime(df["fecha_nacimiento"]).min().date()
    fmax = pd.to_datetime(df["fecha_nacimiento"]).max().date()
    print(f"Rango de fechas de nacimiento: {fmin} a {fmax}")

    print("\nConteo por finca:")
    for finca, conteo in df["finca"].value_counts().items():
        print(f"  {finca:<14} {conteo}")

    print("\nNacimientos por anio:")
    anios = pd.to_datetime(df["fecha_nacimiento"]).dt.year
    for anio, conteo in anios.value_counts().sort_index().items():
        print(f"  {anio}: {conteo}")

    print("\nPorcentaje de nulos por columna:")
    nulos = df.isna().mean().mul(100).round(1)
    for col, pct in nulos.items():
        print(f"  {col:<28} {pct:>5}%")

    print("\nPeso de destete promedio por finca:")
    prom = df.groupby("finca")["peso_destete"].mean().round(1)
    for finca, valor in prom.items():
        print(f"  {finca:<14} {valor} kg")
    print("=" * 60)


def main() -> None:
    df = generate_dataset()
    _validar_coherencia(df)

    # Asegurar que exista el directorio de salida
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Dataset guardado en: {OUTPUT_PATH}\n")
    imprimir_resumen(df)


if __name__ == "__main__":
    main()
