# MobileNetV2 para visión

La evolución avanzada del showcase puede usar **MobileNetV2**, una arquitectura convolucional ligera preentrenada en ImageNet. Es apropiada para una demostración porque tiene menor costo computacional que arquitecturas más grandes.

Referencia oficial:

- Torchvision: https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html
- Pesos preentrenados: `MobileNet_V2_Weights.DEFAULT`
- Dataset de referencia: ImageNet-1K

Flujo:

```text
Imagen → resize/normalización → MobileNetV2 → top-5 clases → confianza
```

En el modo base del showcase no se descargan pesos durante la clase: se usa un baseline offline para garantizar reproducibilidad. El modo avanzado se habilita explícitamente con `VISION_PRETRAINED=true`, tras instalar `torch` y `torchvision` y descargar/verificar los pesos.

Limitaciones:

- ImageNet no representa todos los contextos reales.
- La confianza no equivale a exactitud.
- El modelo puede heredar sesgos del dataset.
- No debe usarse para decisiones críticas sin validación adicional.

No se suben pesos grandes ni credenciales al repositorio sin definir previamente una estrategia de artefactos.