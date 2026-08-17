"""Streamlit AI Lab: educational showcase of prediction, vision and RAG."""
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.tabular import FEATURES, build_demo_dataset, dataframe_from_csv, predict_tabular, evaluate_tabular, train_tabular_model
from src.explainability import global_importance, local_explanation
from src.vision import classify_image, load_image
from src.rag import LocalRAG, load_documents
from src.model_registry import load_model_artifact
from src.chatbot import answer_controlled, chatbot_status

st.set_page_config(page_title="Streamlit AI Lab", page_icon="🤖", layout="wide")
st.title("🧪 Streamlit AI Lab")
st.caption("Showcase educativo: modelo tabular explicable · visión artificial · RAG local")

@st.cache_data
def get_data():
    return build_demo_dataset()

@st.cache_resource
def get_model():
    return load_model_artifact()

@st.cache_resource
def get_rag():
    return LocalRAG(load_documents(ROOT / "documents"))

with st.sidebar:
    st.header("Laboratorio")
    st.info("Este demo funciona sin API keys ni descargas en vivo.")
    st.markdown("**Flujo:** datos → modelo → predicción → explicación → interacción")
    st.download_button("Descargar datos demo", get_data().drop(columns=["renovara"]).to_csv(index=False), "datos_demo.csv", "text/csv")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Predicción + XAI", "🖼️ Visión", "💬 RAG", "🏗️ Arquitectura"])

with tab1:
    st.header("Predicción tabular explicable")
    model, model_info = get_model()
    _, X_train, X_test, _, y_test = train_tabular_model(get_data())
    st.caption(f"Modelo: {model_info['source']} · artefacto serializado: {'sí' if model_info['artifact'] else 'fallback local'}")
    metric_cols = st.columns(3)
    metrics = evaluate_tabular(model, X_test, y_test)
    metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
    metric_cols[1].metric("F1", f"{metrics['f1']:.1%}")
    metric_cols[2].metric("AUC", f"{metrics['auc']:.1%}")
    st.divider()
    mode = st.radio("Modo de predicción", ["Inputs manuales", "Carga masiva CSV"], horizontal=True)
    if mode == "Inputs manuales":
        with st.form("manual_form"):
            cols = st.columns(5)
            values = {}
            defaults = [1200.0, 18, 3, 8, 10]
            for col, feature, default in zip(cols, FEATURES, defaults):
                values[feature] = col.number_input(feature, value=float(default), min_value=0.0)
            submitted = st.form_submit_button("Predecir")
        if submitted:
            row = pd.DataFrame([values])
            result = predict_tabular(model, row)
            probability = float(result.iloc[0]["probabilidad_renovacion"])
            st.success(f"Predicción: {'renovará' if probability >= .5 else 'no renovará'} · probabilidad {probability:.1%}")
            explanation, method = local_explanation(model, row, X_train)
            st.subheader(f"Explicación local — {method}")
            st.bar_chart(explanation.set_index("feature")["impact"])
            st.dataframe(explanation, use_container_width=True, hide_index=True)
    else:
        st.caption("Columnas requeridas: " + ", ".join(FEATURES))
        uploaded = st.file_uploader("Sube un CSV", type="csv")
        if uploaded:
            try:
                frame = dataframe_from_csv(uploaded.getvalue())
                result = predict_tabular(model, frame)
                st.dataframe(result, use_container_width=True)
                st.download_button("Descargar predicciones", result.to_csv(index=False), "predicciones.csv", "text/csv")
            except ValueError as exc:
                st.error(str(exc))
    st.subheader("Importancia global")
    importance, method = global_importance(model, X_test, y_test)
    st.caption(f"Método mostrado: {method}. SHAP/LIME se activan automáticamente si están instalados.")
    st.bar_chart(importance.set_index("feature")["importance"])

with tab2:
    st.header("Visión artificial")
    st.write("Clasificación de imágenes integrada con MobileNetV2 (PyTorch) y fallback didáctico offline.")
    uploaded = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"], key="vision")
    force_base = st.checkbox("Usar baseline heurístico didáctico", value=False)
    if uploaded:
        image = load_image(uploaded.getvalue())
        st.image(image, caption="Imagen recibida", width=320)
        result = classify_image(image, force_baseline=force_base)
        model_name = result.get("model_name", "Visión artificial")
        st.caption(f"Modelo activo: **{model_name}**")
        st.success(f"Clase estimada: {result['label']} · confianza {result['confidence']:.1%}")
        st.bar_chart(pd.Series(result["probabilities"], name="probabilidad"))
        st.warning("Este modelo es con fines de demostración y no debe usarse para decisiones críticas sin validación adicional.")
    else:
        st.info("Sube una imagen para ejecutar el flujo de visión.")

with tab3:
    st.header("RAG conversacional local")
    st.write("Recupera fragmentos de los documentos del repositorio y muestra las fuentes. No depende de una API externa.")
    question = st.text_input("Pregunta sobre el showcase", placeholder="¿Cómo se ejecuta la aplicación?")
    if question:
        response = answer_controlled(get_rag(), question)
        st.markdown(response["answer"])
        st.caption(f"Modo: {response['mode']} · proveedor externo: {response['provider']}")
        if response["sources"]:
            st.subheader("Fuentes recuperadas")
            for source in response["sources"]:
                st.caption(f"{source['source']} · score {source['score']}")
                st.code(source["excerpt"])
        else:
            st.warning("Respuesta no fundamentada: prueba otra pregunta basada en los documentos.")

with tab4:
    st.header("Arquitectura educativa")
    st.code("Usuario → Streamlit → [Modelo tabular + SHAP/LIME]\n                  → [Visión artificial]\n                  → [Retriever RAG + documentos]", language="text")
    st.markdown("""
    **Tres niveles de madurez:**
    1. Predicción: el modelo produce una salida.
    2. Explicabilidad: SHAP/LIME ayudan a entenderla.
    3. Sistema aplicado: una interfaz integra modelos, documentos y usuario.
    """)
    st.info("La interfaz no es el modelo: es la capa que convierte el modelo en una solución demostrable y utilizable.")

st.divider()
st.caption("Streamlit AI Lab · Showcase educativo · versión 1.0.0")
