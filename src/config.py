"""
Carga de variables de entorno para el agente ganadero.

Lee la configuracion desde un archivo .env (via python-dotenv) y expone las
constantes que usa el resto de la aplicacion. Falla temprano y con un mensaje
claro si falta la API key de Google.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Cargar el archivo .env desde la raiz del proyecto (si existe)
load_dotenv()


def _get_required(name: str) -> str:
    # Obtener una variable obligatoria o fallar con instrucciones claras
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Error: La variable {name} no está configurada. "
            "Copia .env.example a .env"
        )
    return value


# API key de Google (obligatoria)
GOOGLE_API_KEY: str = _get_required("GOOGLE_API_KEY")

# Modelo de Gemini a utilizar
MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")

# Ruta al dataset ganadero
DATASET_PATH: str = os.getenv("DATASET_PATH", "data/ganaderia_dataset.csv")

# Temperatura del modelo (0 = respuestas deterministas)
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))
