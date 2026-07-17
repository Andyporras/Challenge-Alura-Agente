"""
Construccion del agente LangChain + Gemini para el dataset ganadero.

Crea un agente de pandas que traduce preguntas en lenguaje natural a codigo
sobre el dataframe, usando Gemini como modelo de lenguaje.
"""

from __future__ import annotations

import pandas as pd
from langchain.agents import AgentExecutor
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GOOGLE_API_KEY, MODEL_NAME, TEMPERATURE
from src.data_loader import get_dataset_summary
from src.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT

# Limite de iteraciones para evitar loops del agente
MAX_ITERATIONS = 8


def build_agent(df: pd.DataFrame) -> AgentExecutor:
    """Construye el agente de pandas con Gemini como LLM.

    Inyecta el system prompt, los ejemplos few-shot y un resumen del dataset
    como prefix, de modo que el agente conozca la estructura de los datos
    antes de ejecutar codigo.

    Args:
        df: DataFrame ganadero ya cargado y validado.

    Returns:
        AgentExecutor listo para recibir preguntas.
    """
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )

    # Prefix completo: instrucciones + ejemplos + estructura del dataset
    prefix = (
        f"{SYSTEM_PROMPT}\n"
        f"{FEW_SHOT_EXAMPLES}\n"
        "Resumen del dataset con el que trabajas:\n"
        f"{get_dataset_summary(df)}"
    )

    # tool-calling usa function calling nativo de Gemini, mucho mas
    # confiable que el parseo de texto del formato ReAct por defecto
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        prefix=prefix,
        agent_type="tool-calling",
        verbose=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
        max_iterations=MAX_ITERATIONS,
        allow_dangerous_code=True,  # requerido por el agente de pandas
    )

    return agent


def ask(agent: AgentExecutor, question: str) -> str:
    """Ejecuta una pregunta contra el agente y devuelve la respuesta.

    Captura cualquier excepcion del agente para no romper la interfaz y
    devolver un mensaje amigable al usuario.

    Args:
        agent: AgentExecutor construido con build_agent.
        question: Pregunta en lenguaje natural.

    Returns:
        Respuesta del agente en texto, o un mensaje de error amigable.
    """
    try:
        # Reintentar una vez si el modelo devuelve una respuesta vacia
        for _ in range(2):
            result = agent.invoke({"input": question})
            answer = _coerce_output(result.get("output", ""))
            if answer.strip():
                return answer
        return "Error: No pude procesar la pregunta. Intenta reformularla."
    except Exception:
        return "Error: No pude procesar la pregunta. Intenta reformularla."


def _coerce_output(output: object) -> str:
    # Gemini puede devolver la respuesta como lista de partes de contenido
    if isinstance(output, list):
        parts = []
        for part in output:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(output)
