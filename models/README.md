# Artefactos de modelos

El showcase carga un modelo tabular serializado desde `models/random_forest_renewal.joblib` cuando el artefacto está presente y validado. Si falta, la aplicación entrena el modelo determinísticamente para no bloquear el demo.

El archivo binario no se versiona por defecto en GitHub porque los artefactos binarios pueden crecer rápidamente. Para un repositorio de demo se recomienda:

- Git LFS; o
- GitHub Releases; o
- un almacenamiento de artefactos con URL y hash SHA-256.

Generar el artefacto localmente:

```bash
python scripts/train_model.py
```

Verificarlo:

```bash
python scripts/verify_model.py
```

MobileNetV2 se descarga/cacha mediante Torchvision únicamente si `VISION_PRETRAINED=true`. Su referencia oficial se documenta en `documents/mobilenet.md`. Mantener siempre disponible el baseline offline para la clase.

No subir claves, `.env` ni datos privados.

> Este directorio documenta dónde vive el modelo serializado; la aplicación muestra en pantalla si cargó el artefacto o si usó el fallback de entrenamiento.
> 
> Antes de publicar, elegir explícitamente entre Git LFS, Release o almacenamiento externo para que una clonación limpia pueda obtener el artefacto.