# Checklist antes del demo en vivo

## Instalación

- [ ] Crear entorno virtual limpio.
- [ ] Instalar `requirements.txt`.
- [ ] Ejecutar `pytest -q`.
- [ ] Ejecutar `python -m compileall app.py src`.
- [ ] Ejecutar `streamlit run app.py`.

## Recorrido funcional

- [ ] Tabular: predicción manual.
- [ ] Tabular: CSV masivo válido.
- [ ] Tabular: CSV inválido y mensaje de error.
- [ ] Tabular: importancia global y explicación local.
- [ ] Tabular: indicador de artefacto serializado y metadatos.
- [ ] Visión: imagen válida.
- [ ] RAG: pregunta con evidencia.
- [ ] RAG: pregunta sin evidencia.
- [ ] RAG: confirmar que el modo mostrado es `local-grounded`.
- [ ] Descargar datos y predicciones.

## Contingencia

- [ ] Tener el repositorio clonado localmente.
- [ ] Tener capturas o video de respaldo.
- [ ] No depender de API keys.
- [ ] Verificar que las imágenes de prueba estén disponibles.
- [ ] Abrir el README antes de comenzar.

## Narrativa

1. La interfaz recibe una entrada.
2. El componente ejecuta un modelo o recupera contexto.
3. La salida se presenta con explicación o fuentes.
4. El usuario revisa límites y toma una decisión informada.

## Evidencia para mostrar

- URL de GitHub.
- Ruta del artefacto `models/random_forest_renewal.joblib` y su SHA-256.
- README.
- Estructura del repositorio.
- Terminal con `streamlit run app.py`.
- Aplicación funcionando.
- Tests exitosos.
- Limitaciones documentadas.

> Si una parte avanzada falla, volver al modo offline base y explicar honestamente qué componente se está demostrando.

## Nota sobre el modo offline

El showcase base deliberadamente evita descargar pesos o llamar APIs durante la clase. Esto permite reproducir la experiencia en un entorno limpio y separar el concepto didáctico de la complejidad de producción.

> Antes de publicar, revisar dependencias, licencias y cualquier dato que no sea sintético o público.

## Limitaciones conocidas del showcase

- Visión usa un baseline heurístico, no un clasificador real de animales.
- RAG usa recuperación léxica y no generación por LLM.
- SHAP/LIME son opcionales en el entorno base.

Estas limitaciones son intencionales para garantizar un demo estable; deben explicarse durante la clase y no ocultarse.