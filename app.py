"""Streamlit AI Lab: Apple-inspired educational showcase of tabular AI, vision & RAG."""
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
from src.chatbot import answer_controlled
from src.ui_components import (
    inject_apple_theme,
    create_global_importance_chart,
    create_local_explanation_chart,
    create_vision_probabilities_chart,
    create_donut_gauge,
)

st.set_page_config(page_title="Streamlit AI Lab", page_icon="⚡", layout="wide")

# Inject Apple-inspired Design System
inject_apple_theme()


@st.cache_data
def get_data():
    return build_demo_dataset()


@st.cache_resource
def get_model():
    return load_model_artifact()


@st.cache_resource
def get_rag():
    return LocalRAG(load_documents(ROOT / "documents"))


# --- Hero Header ---
st.markdown(
    """
    <div class="hero-banner">
        <h1 class="hero-title">Streamlit AI Lab</h1>
        <p class="hero-subtitle">Plataforma educativa de Inteligencia Aplicada: Modelos Tabulares Explicables, Visión por Computadora y RAG Local Privado.</p>
        <div style="margin-top: 14px;">
            <span class="status-badge status-success">● Random Forest Activo</span>
            <span class="status-badge status-info">● PyTorch MobileNetV2</span>
            <span class="status-badge status-warning">● RAG Local 100% Offline</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚡ Control Center")
    st.caption("Laboratorio de IA demostrativo 100% ejecutable sin credenciales ni APIs externas.")

    st.markdown("---")
    st.markdown("**Arquitectura:**")
    st.markdown("- **Tabular:** Random Forest + XAI\n- **Visión:** PyTorch MobileNetV2\n- **RAG:** Búsqueda Documental Local")

    st.markdown("---")
    st.download_button(
        "📥 Descargar Dataset Demo (CSV)",
        get_data().drop(columns=["renovara"]).to_csv(index=False),
        "datos_demo_renovacion.csv",
        "text/csv",
        use_container_width=True,
    )

# --- Navigation Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Predicción & XAI",
    "🖼️ Visión Artificial",
    "💬 RAG Conversacional",
    "🏗️ Arquitectura",
])

# ==========================================
# TAB 1: PREDICCIÓN TABULAR + EXPLICABILIDAD
# ==========================================
with tab1:
    st.markdown("## 📊 Predicción Tabular & Explicabilidad (XAI)")
    st.caption("Evalúa el rendimiento global del modelo de renovación y genera explicaciones locales e importancias globales.")

    model, model_info = get_model()
    _, X_train, X_test, _, y_test = train_tabular_model(get_data())
    metrics = evaluate_tabular(model, X_test, y_test)

    # Apple Metric Widgets
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{metrics['accuracy']:.1%}</div>
                <div class="metric-lbl">Accuracy</div>
                <div class="metric-sub">Exactitud general</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{metrics['f1']:.1%}</div>
                <div class="metric-lbl">F1 Score</div>
                <div class="metric-sub">Balance precisión/recall</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{metrics['auc']:.1%}</div>
                <div class="metric-lbl">AUC-ROC</div>
                <div class="metric-sub">Capacidad de discriminación</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{len(get_data())}</div>
                <div class="metric-lbl">Muestras</div>
                <div class="metric-sub">Dataset sintético</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    mode = st.radio("Modo de entrada de datos:", ["Entrada Manual Interactiva", "Predicción Masiva por CSV"], horizontal=True)

    if mode == "Entrada Manual Interactiva":
        with st.form("manual_form"):
            st.markdown("##### ⚙️ Ingrese los valores del cliente")
            cols = st.columns(5)
            values = {}
            defaults = [1200.0, 18, 3, 8, 10]
            labels = ["Ingreso Mensual ($)", "Antigüedad (Meses)", "Tickets Previos", "Satisfacción (1-10)", "Tiempo Resp. (días)"]

            for col, feature, default, label in zip(cols, FEATURES, defaults, labels):
                values[feature] = col.number_input(label, value=float(default), min_value=0.0)

            submitted = st.form_submit_button("🔮 Predecir & Explicar", use_container_width=True)

        if submitted:
            row = pd.DataFrame([values])
            result = predict_tabular(model, row)
            probability = float(result.iloc[0]["probabilidad_renovacion"])
            prediction_label = "Renovará" if probability >= 0.5 else "No Renovará"

            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.altair_chart(create_donut_gauge(probability, "Probabilidad de Renovación"), use_container_width=True)
            with res_col2:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <h3 style="margin:0; color:{'#34D399' if probability >= 0.5 else '#F87171'}; font-size:1.8rem;">
                            Predicción: {prediction_label}
                        </h3>
                        <p style="color:#94A3B8; font-size:1.1rem; margin-top:6px;">
                            Probabilidad estimada: <strong>{probability:.1%}</strong>
                        </p>
                        <p style="color:#64748B; font-size:0.9rem;">
                            Cálculo generado con RandomForestClassifier basado en el artefacto serializado.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            explanation, method = local_explanation(model, row, X_train)
            st.markdown(f"### 🔬 Explicación Local del Resultado ({method})")

            exp_col1, exp_col2 = st.columns([1.5, 1])
            with exp_col1:
                st.altair_chart(create_local_explanation_chart(explanation, method), use_container_width=True)
            with exp_col2:
                st.dataframe(explanation, use_container_width=True, hide_index=True)
    else:
        st.markdown("##### 📁 Carga masiva de clientes mediante CSV")
        st.caption("Columnas requeridas: `" + "`, `" .join(FEATURES) + "`")
        uploaded = st.file_uploader("Subir archivo CSV", type="csv")
        if uploaded:
            try:
                frame = dataframe_from_csv(uploaded.getvalue())
                result = predict_tabular(model, frame)
                st.dataframe(result, use_container_width=True)
                st.download_button("📥 Descargar Resultados (CSV)", result.to_csv(index=False), "predicciones_masivas.csv", "text/csv")
            except ValueError as exc:
                st.error(str(exc))

    st.markdown("---")
    st.markdown("### 🌐 Importancia Global de Características")
    importance, method = global_importance(model, X_test, y_test)
    st.caption(f"Método utilizado: **{method}**. Muestra el peso general que asigna el modelo a cada variable.")
    st.altair_chart(create_global_importance_chart(importance, method), use_container_width=True)

# ==========================================
# TAB 2: VISIÓN ARTIFICIAL
# ==========================================
with tab2:
    st.markdown("## 🖼️ Visión Artificial con PyTorch")
    st.caption("Clasificación de imágenes con pesos preentrenados de MobileNetV2 (ImageNet) y fallback didáctico offline.")

    vis_col1, vis_col2 = st.columns([1, 1.2])

    with vis_col1:
        uploaded_img = st.file_uploader("Cargar una imagen", type=["png", "jpg", "jpeg"], key="vision_uploader")
        force_base = st.checkbox("Usar baseline heurístico didáctico", value=False)
        if uploaded_img:
            image = load_image(uploaded_img.getvalue())
            st.image(image, caption="Imagen cargada para inferencia", use_container_width=True)
        else:
            st.info("💡 Sube cualquier imagen (ej. animal, vehículo, objeto) para clasificarla en tiempo real.")

    with vis_col2:
        if uploaded_img:
            result = classify_image(image, force_baseline=force_base)
            model_name = result.get("model_name", "Visión Artificial")

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="status-badge status-info" style="margin-bottom:10px;">● {model_name}</div>
                    <h2 style="margin:0; color:#F8FAFC;">{result['label'].capitalize()}</h2>
                    <p style="color:#60A5FA; font-size:1.2rem; font-weight:700; margin-top:4px;">
                        Confianza: {result['confidence']:.1%}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.altair_chart(create_vision_probabilities_chart(result["probabilities"], result["label"]), use_container_width=True)
            st.warning("⚠️ Nota: Este módulo demuestra la inferencia visual en tiempo real. La confianza representa el cálculo de distribución softmax del modelo.")

# ==========================================
# TAB 3: RAG CONVERSACIONAL
# ==========================================
with tab3:
    st.markdown("## 💬 RAG Conversacional Local & Privado")
    st.caption("Recupera fragmentos fundamentados de la base de conocimientos documental en `documents/` y muestra citas exactas.")

    st.markdown("##### 💡 Preguntas sugeridas (haz clic para probar):")
    q_cols = st.columns(4)
    suggested_q = None
    if q_cols[0].button("¿Cómo se ejecuta la app?", use_container_width=True):
        suggested_q = "¿Cómo se ejecuta la aplicación?"
    if q_cols[1].button("¿Qué es SHAP?", use_container_width=True):
        suggested_q = "¿Qué es SHAP?"
    if q_cols[2].button("¿Qué es RAG?", use_container_width=True):
        suggested_q = "¿Qué significa RAG?"
    if q_cols[3].button("¿Qué es MobileNetV2?", use_container_width=True):
        suggested_q = "¿Qué es MobileNetV2?"

    question = st.text_input("Ingresa tu pregunta:", value=suggested_q if suggested_q else "", placeholder="Ej: ¿Cómo se explican los modelos?")

    if question:
        response = answer_controlled(get_rag(), question)

        st.markdown(
            f"""
            <div class="result-card">
                <div style="margin-bottom:8px;">
                    <span class="status-badge status-success">Modo: {response['mode']}</span>
                    <span class="status-badge status-info">Proveedor externo: {response['provider']}</span>
                </div>
                <div style="font-size:1.1rem; color:#F8FAFC; line-height:1.6;">
                    {response['answer']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if response["sources"]:
            st.markdown("### 📚 Fuentes & Citas Recuperadas")
            for idx, source in enumerate(response["sources"], 1):
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <strong style="color:#60A5FA;">Fuente #{idx}: {source['source']}</strong>
                            <span class="status-badge status-info">Relevancia Score: {source['score']:.2f}</span>
                        </div>
                        <code style="display:block; color:#CBD5E1; background:rgba(0,0,0,0.3); padding:10px; border-radius:6px; font-size:0.88rem;">
                            {source['excerpt']}
                        </code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("⚠️ Respuesta no fundamentada: No se encontraron fragmentos con suficiente relevancia en la base de documentos.")

# ==========================================
# TAB 4: ARQUITECTURA
# ==========================================
with tab4:
    st.markdown("## 🏗️ Arquitectura Educativa del Sistema")
    st.caption("Diagrama de flujo de datos y componentes integrados en Streamlit AI Lab.")

    st.markdown(
        """
        ```text
        ┌────────────────────────────────────────────────────────────────────────┐
        │                          Streamlit AI Lab                              │
        └──────────────────────────────────┬─────────────────────────────────────┘
                                           │
             ┌─────────────────────────────┼─────────────────────────────┐
             ▼                             ▼                             ▼
        ┌─────────┐                   ┌─────────┐                   ┌─────────┐
        │ Tabular │                   │ Visión  │                   │   RAG   │
        └────┬────┘                   └────┬────┘                   └────┬────┘
             │                             │                             │
             ▼                             ▼                             ▼
        Random Forest                 MobileNetV2                   Búsqueda Léxica
        + SHAP / LIME                 PyTorch ImageNet               + Citas Documentales
        ```
        """
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color:#60A5FA; margin-top:0;">1. Predicción</h4>
                <p style="color:#94A3B8; font-size:0.9rem;">El modelo calcula las probabilidades basadas en patrones aprendidos.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color:#34D399; margin-top:0;">2. Explicabilidad</h4>
                <p style="color:#94A3B8; font-size:0.9rem;">SHAP y LIME desglosan las contribuciones individuales de cada feature.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color:#FBBF24; margin-top:0;">3. Interfaz Aplicada</h4>
                <p style="color:#94A3B8; font-size:0.9rem;">Streamlit transforma modelos abstractos en una solución utilizable.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption("Streamlit AI Lab · Design System inspirado en Apple · Versión 2.0.0")
