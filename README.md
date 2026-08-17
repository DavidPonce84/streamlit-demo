# Streamlit AI Lab ⚡

Showcase educativo e interactivo con interfaz inspirada en el **Design System de Apple (macOS/iOS)** para demostrar cómo una aplicación de Inteligencia Aplicada integra:

1. **Predicción Tabular & XAI:** Modelo Random Forest explicable, métricas globales (Accuracy, F1, AUC), predicción individual con medidor Donut Gauge, predicción masiva por CSV y gráficos bidireccionales de impacto de características (SHAP/LIME o fallback).
2. **Visión Artificial en Tiempo Real:** Clasificación de imágenes integrada por defecto con **PyTorch y MobileNetV2** (descarga automática y almacenamiento en caché de pesos de ImageNet de 14 MB en la primera ejecución), con opción de alternar al baseline didáctico offline.
3. **RAG / Chatbot Conversacional Local:** Recuperación documental fundamentada en `documents/` con citas de fuentes, puntuación de relevancia y respuestas controladas 100% privadas.
4. **Artefactos & Metadatos del Modelo:**
   - **`models/random_forest_renewal.joblib`**: Objeto binario del modelo de Machine Learning entrenado (`RandomForestClassifier`).
   - **`models/model_metadata.json`**: Ficha técnica en texto plano con variables requeridas, `random_state` y métricas de desempeño.

> **Mensaje didáctico:** Streamlit no es el modelo. Es la capa de diseño y experiencia de usuario que convierte un modelo de IA en una solución demostrable, transparente y utilizable.

---

## Requisitos

- Python 3.10–3.12 recomendado.
- Entorno virtual (`.venv`).
- No requiere credenciales ni conexión a APIs de pago externas.

---

## Estructura Relevante

```text
app.py                                # Interfaz principal Streamlit con Apple Design System
src/
├── ui_components.py                  # Componentes de diseño Apple y gráficos con Altair
├── tabular.py                        # Entrenamiento y predicción del modelo Random Forest
├── explainability.py                 # Explicaciones locales y globales (SHAP/LIME/fallback)
├── vision.py                         # Clasificador PyTorch MobileNetV2 + fallback didáctico
├── rag.py                            # Motor RAG local de recuperación documental
├── chatbot.py                        # Lógica de respuesta fundamentada con citas
└── model_registry.py                 # Carga segura y validación de artefactos y metadatos
models/
├── random_forest_renewal.joblib      # Modelo binario serializado (Random Forest)
└── model_metadata.json              # Ficha técnica y métricas en formato JSON
documents/                            # Base de conocimientos local en Markdown
tests/                                # Suite de 24 pruebas unitarias automatizadas
requirements.txt                      # Dependencias principales (Streamlit, PyTorch, Scikit-Learn, Altair, etc.)
```

---

## Ejecución Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/DavidPonce84/streamlit-demo.git
cd streamlit-demo

# 2. Crear y activar el entorno virtual
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate      # Windows

# 3. Actualizar pip e instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Iniciar la aplicación
streamlit run app.py
```

Abrir la URL local en el navegador: **`http://localhost:8501`**

---

## ¿Cómo Gestiona el Sistema los Modelos?

### 1. PyTorch MobileNetV2 (Visión)
- **Primera ejecución:** Al subir una imagen por primera vez en la Pestaña 2, PyTorch descarga automáticamente los pesos oficiales de ImageNet (~14 MB) desde los servidores de PyTorch.
- **Caché Local:** Se almacenan localmente en `~/.cache/torch/hub/checkpoints/mobilenet_v2-7ebf99e0.pth`.
- **Siguientes ejecuciones:** Se cargan directamente desde el disco sin consumir conexión a internet.

### 2. Modelo Tabular (Random Forest / XGBoost)
- **Random Forest (Incluido):** Se carga desde `models/random_forest_renewal.joblib` validado con `models/model_metadata.json`.
- **Evolución con XGBoost:** Para entrenar o reemplazar con XGBoost, se instala `pip install xgboost`, se entrena con `model.fit(X_train, y_train)` y se guarda localmente mediante `joblib.dump(model, "models/xgboost_renewal.joblib")` o `model.save_model("models/xgboost_renewal.json")`.

---

## Recorrido Recomendado para la Clase / Demo

### Tab 1 — Predicción & XAI
1. Evaluar las métricas generales en las tarjetas Apple Widget (`Accuracy`, `F1`, `AUC-ROC`).
2. Probar la **Entrada Manual Interactiva** y presionar **🔮 Predecir & Explicar**.
3. Observar el **Donut Gauge** de probabilidad y el **Gráfico Bidireccional de Explicabilidad Local** (verde para variables que favorecen la renovación, rojo para las que la desfavorecen).
4. Analizar el gráfico de **Importancia Global de Características**.

### Tab 2 — Visión Artificial
1. Cargar una imagen de prueba (ej. vehículo, animal, objeto).
2. Observar la inferencia en tiempo real con **MobileNetV2 (PyTorch)** y el gráfico de **Top-5 Clases Predichas**.
3. Probar el checkbox *"Usar baseline heurístico didáctico"* para mostrar el modo offline de respaldo.

### Tab 3 — RAG Conversacional Local
1. Probar los botones de **Preguntas Sugeridas** (*¿Cómo se ejecuta la app?*, *¿Qué es SHAP?*, *¿Qué es MobileNetV2?*).
2. Observar la respuesta fundamentada y las **Fuentes & Citas Documentales** con su puntuación de relevancia.

---

## Pruebas Automatizadas

```bash
pytest -v
```
*(Valida los 24 tests unitarios de la aplicación).*

---

## Licencia

Material educativo para demostración de arquitectura de Inteligencia Aplicada.