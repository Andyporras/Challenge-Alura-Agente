"""
Construccion del agente LangChain + Gemini para el dataset ganadero.

Crea un agente con una tool de pandas que traduce preguntas en lenguaje
natural a codigo sobre el dataframe, usando Gemini como modelo de lenguaje.
Usa langchain 1.x (create_agent sobre LangGraph) con soporte completo para
los modelos Gemini 3.x.
"""

from __future__ import annotations

import ast
import contextlib
import io
from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GOOGLE_API_KEY, MODEL_NAME, TEMPERATURE
from src.data_loader import get_dataset_summary
from src.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT

# Limite de pasos del agente para evitar loops
MAX_ITERATIONS = 8


def _run_pandas_code(code: str, namespace: dict[str, Any]) -> str:
    # Ejecutar el codigo y devolver stdout mas el valor de la ultima
    # expresion, imitando el comportamiento de un REPL
    tree = ast.parse(code)
    stdout = io.StringIO()
    value: Any = None

    with contextlib.redirect_stdout(stdout):
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            body = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(body, "<agente>", "exec"), namespace)
            last_expr = ast.Expression(body=tree.body[-1].value)
            value = eval(compile(last_expr, "<agente>", "eval"), namespace)
        else:
            exec(compile(tree, "<agente>", "exec"), namespace)

    output = stdout.getvalue()
    if value is not None:
        output += repr(value)
    return output if output.strip() else "(el código no produjo salida)"


def build_agent(df: pd.DataFrame) -> Runnable:
    """Construye el agente de pandas con Gemini como LLM.

    Inyecta el system prompt, los ejemplos few-shot y un resumen del dataset,
    de modo que el agente conozca la estructura de los datos antes de
    ejecutar codigo.

    Args:
        df: DataFrame ganadero ya cargado y validado.

    Returns:
        Agente LangGraph listo para recibir preguntas via ask().
    """
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )

    # Namespace compartido entre invocaciones de la tool: permite que el
    # agente defina variables en un paso y las reuse en el siguiente
    namespace: dict[str, Any] = {"df": df, "pd": pd}

    @tool
    def python_repl_ast(query: str) -> str:
        """Ejecuta código Python sobre el dataframe `df` (pandas ya está
        importado como `pd`) y devuelve la salida. Úsalo para calcular
        cualquier dato que necesites para responder."""
        try:
            return _run_pandas_code(query, namespace)
        except Exception as exc:
            return f"Error al ejecutar el código: {exc}"

    # System prompt completo: instrucciones + ejemplos + estructura de datos
    system_prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"{FEW_SHOT_EXAMPLES}\n"
        "Resumen del dataset con el que trabajas:\n"
        f"{get_dataset_summary(df)}"
    )

    return create_agent(
        model=llm,
        tools=[python_repl_ast],
        system_prompt=system_prompt,
    )


def ask(agent: Runnable, question: str) -> str:
    """Ejecuta una pregunta contra el agente y devuelve la respuesta.

    Captura cualquier excepcion del agente para no romper la interfaz y
    devolver un mensaje amigable al usuario.

    Args:
        agent: Agente construido con build_agent.
        question: Pregunta en lenguaje natural.

    Returns:
        Respuesta del agente en texto, o un mensaje de error amigable.
    """
    config = {"recursion_limit": 2 * MAX_ITERATIONS + 1}
    try:
        # Reintentar una vez si el modelo devuelve una respuesta vacia
        for _ in range(2):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )
            messages = result.get("messages", [])
            answer = _coerce_output(messages[-1].content) if messages else ""
            if answer.strip():
                return answer
        return "Error: No pude procesar la pregunta. Intenta reformularla."
    except Exception as exc:
        # Distinguir el limite de cuota de la API de otros errores
        if _is_quota_error(exc):
            return (
                "Error: Se alcanzó el límite de uso de la API de Gemini. "
                "Intenta de nuevo más tarde."
            )
        return "Error: No pude procesar la pregunta. Intenta reformularla."


def _is_quota_error(exc: Exception) -> bool:
    # Detectar errores 429 / ResourceExhausted sin depender de la clase exacta
    text = f"{type(exc).__name__} {exc}".lower()
    return "429" in text or "resourceexhausted" in text or "quota" in text


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
