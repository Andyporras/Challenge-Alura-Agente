"""
Interfaz Streamlit del agente ganadero.

Chat en lenguaje natural sobre el dataset de gestion ganadera, con
historial de conversacion, preguntas de ejemplo y vista de datos crudos.

Uso:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_loader import load_dataset, validate_dataset

st.set_page_config(
    page_title="Agente Ganadero",
    page_icon="🐄",
    layout="wide",
)

# Preguntas de ejemplo para el sidebar
EXAMPLE_QUESTIONS = [
    "¿Cuántos animales hay registrados en total?",
    "¿Cuál es el peso promedio al destete por finca?",
    "¿Qué finca tuvo más nacimientos en 2024?",
    "¿Cuál toro tiene la mayor cantidad de crías registradas?",
    "¿Cuál es el costo promedio de alimentación mensual por finca?",
]

# La importacion de config falla si no hay API key: mostrar instrucciones
try:
    from src.config import DATASET_PATH
except EnvironmentError as exc:
    st.error(
        f"{exc}\n\n"
        "Pasos: copia `.env.example` a `.env` y coloca tu API key de "
        "Google AI Studio en la variable `GOOGLE_API_KEY`. "
        "Puedes obtenerla en https://aistudio.google.com/apikey"
    )
    st.stop()


@st.cache_data
def get_dataframe(path: str) -> pd.DataFrame:
    # Cargar y validar el dataset una sola vez
    df = load_dataset(path)
    validate_dataset(df)
    return df


@st.cache_resource
def get_agent(path: str):
    # Construir el agente una sola vez por sesion del servidor
    from src.agent import build_agent

    return build_agent(get_dataframe(path))


def render_sidebar(df: pd.DataFrame) -> None:
    with st.sidebar:
        st.header("Sobre el proyecto")
        st.markdown(
            "Asistente inteligente para gestión ganadera. Haz preguntas en "
            "lenguaje natural sobre partos, pesos, fincas, razas y costos; "
            "el agente (LangChain + Gemini) consulta los datos y responde "
            "en español."
        )

        st.divider()
        st.subheader("Métricas del dataset")
        col1, col2 = st.columns(2)
        col1.metric("Animales", len(df))
        col2.metric("Fincas", df["finca"].nunique())
        fmin = df["fecha_nacimiento"].min().date()
        fmax = df["fecha_nacimiento"].max().date()
        st.caption(f"Nacimientos entre {fmin} y {fmax}")

        st.divider()
        st.subheader("Preguntas de ejemplo")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True):
                st.session_state["pending_question"] = question


def main() -> None:
    # Ocultar el menu interno de Streamlit (solo existe en ingles) para
    # que la interfaz quede completamente en espanol de cara al usuario
    st.markdown(
        "<style>[data-testid='stMainMenu'] {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )

    st.title("Agente Ganadero — Consulta tus datos en lenguaje natural")

    try:
        df = get_dataframe(DATASET_PATH)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar(df)

    with st.expander("Ver datos crudos"):
        st.dataframe(df, use_container_width=True)

    # Historial de chat en session_state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada: chat_input o pregunta de ejemplo clickeada
    question = st.chat_input("Escribe tu pregunta sobre el hato...")
    if pending := st.session_state.pop("pending_question", None):
        question = pending

    if question:
        st.session_state["messages"].append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Consultando los datos..."):
                from src.agent import ask

                answer = ask(get_agent(DATASET_PATH), question)
            st.markdown(answer)

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer}
        )


if __name__ == "__main__":
    main()
