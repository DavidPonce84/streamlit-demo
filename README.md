# Streamlit AI Lab

Showcase educativo para demostrar cómo una aplicación Streamlit integra:

1. **Predicción tabular:** Random Forest, predicción individual y masiva.
2. **Explicabilidad:** importancia global y explicación local con SHAP/LIME cuando están instalados; fallback transparente cuando no lo están.
3. **Visión:** flujo offline de imagen → predicción → probabilidades, con baseline didáctico seguro para una clase en vivo.
4. **RAG/chatbot controlado:** recuperación documental con fuentes y fallback local seguro, sin API keys.
5. **Artefacto serializado:** el Random Forest preentrenado vive en `models/random_forest_renewal.joblib` cuando se versiona con la estrategia elegida.
6. **Visión avanzada opcional:** referencia documentada a MobileNetV2, manteniendo baseline offline para el demo estable.

> **Mensaje didáctico:** Streamlit no es el modelo. Es la capa que convierte un modelo en una solución demostrable y utilizable.

## Requisitos

- Python 3.10–3.12 recomendado.
- Entorno virtual.
- No se necesitan credenciales ni conexión a APIs externas para el demo base.

## Estructura relevante

```text
app.py
models/random_forest_renewal.joblib
models/model_metadata.json
src/model_registry.py
src/chatbot.py
documents/
tests/
```

La aplicación indica en pantalla si cargó el artefacto serializado o si activó el fallback determinístico.

## Ejecución local

```bash
git clone <URL_DEL_REPOSITORIO>
cd streamlit-ai-lab
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate      # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Abrir la URL local que muestra Streamlit, normalmente `http://localhost:8501`.

## Recorrido recomendado para la clase

### Tab 1 — Predicción + XAI

1. Mostrar métricas del Random Forest.
2. Ingresar datos manualmente.
3. Explicar la probabilidad predicha.
4. Interpretar la importancia global.
5. Mostrar explicación local.
6. Subir un CSV y descargar predicciones masivas.

CSV mínimo:

```csv
ingreso_mensual,antiguedad_meses,tickets_previos,satisfaccion,tiempo_respuesta
1200,18,3,8,10
```

### Tab 2 — Visión

Subir una imagen y observar la predicción, probabilidades y advertencia de limitaciones. El baseline incluido es intencionalmente offline y didáctico; el modo avanzado MobileNetV2 está documentado, pero no se activa por defecto para evitar descargas durante el demo.

### Tab 3 — RAG/chatbot controlado

Preguntas sugeridas:

- `¿Cómo se ejecuta la aplicación?`
- `¿Qué es SHAP?`
- `¿Qué significa RAG?`
- `¿Cómo se explican los modelos?`

Después realizar una pregunta que no esté documentada para mostrar la respuesta de insuficiencia de evidencia. La configuración `.env` es opt-in; el modo base no llama proveedores externos.

## Arquitectura

```text
Usuario
  ↓
Streamlit
  ├── src/tabular.py          → Random Forest + CSV
  ├── src/explainability.py   → SHAP/LIME/fallback
  ├── src/vision.py           → baseline de imagen offline
  └── src/rag.py              → recuperación local + fuentes
```

## Reproducibilidad

- Semilla fija: `random_state=42`.
- Dataset sintético generado localmente.
- Documentos RAG versionados en `documents/`.
- No descarga modelos durante la ejecución.
- Dependencias principales versionadas.
- Tests automatizados en `tests/`.

## Limitaciones honestas

- Los datos tabulares son sintéticos y sirven para demostración.
- SHAP y LIME son opcionales; sin ellos se muestran fallbacks explícitos.
- El clasificador de visión incluido es un baseline heurístico offline, no un modelo de visión de producción.
- El RAG utiliza coincidencia léxica local, no embeddings ni un LLM generativo.
- Una explicación no demuestra causalidad y una confianza alta no garantiza exactitud.

## Evolución avanzada opcional

Una segunda versión puede activar:

- XGBoost.
- MobileNetV2/ResNet con pesos predescargados.
- Embeddings vectoriales.
- LLM local con Ollama.
- API externa mediante variables de entorno.

Estas extensiones deben conservar el modo base offline para que el demo no dependa de la red.

## Tests

```bash
pytest -q
```

## Licencia

Material educativo. Añadir una licencia específica antes de publicar el repositorio institucionalmente.

## Demo de respaldo

Antes de una clase en vivo, ejecutar el checklist de `docs/live_demo_checklist.md` y conservar capturas de la aplicación funcionando.