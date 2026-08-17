# Streamlit en el showcase

La aplicación se inicia con `streamlit run app.py`.

Un MVP funcional recibe una entrada, ejecuta un modelo y presenta una salida interpretable. La interfaz no reemplaza al modelo: facilita su uso, evaluación y comunicación.

## Flujo

1. El usuario ingresa datos o una imagen.
2. Streamlit valida la entrada.
3. El componente ejecuta el modelo o recupera documentos.
4. La aplicación presenta resultado, confianza, explicación y fuentes.

## Buenas prácticas

- Mantener dependencias versionadas.
- Evitar credenciales en GitHub.
- Mostrar limitaciones.
- Probar entradas válidas e inválidas.
