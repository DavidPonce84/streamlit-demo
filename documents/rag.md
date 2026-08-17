# RAG local

RAG significa Retrieval-Augmented Generation. Un sistema RAG recupera fragmentos relevantes de una colección documental antes de construir una respuesta.

Flujo básico:

1. Recibir una pregunta.
2. Buscar documentos relacionados.
3. Mostrar el contexto recuperado.
4. Responder con fuentes.
5. Declarar cuando no existe evidencia suficiente.

Este showcase usa recuperación léxica local para ser reproducible y no depender de APIs externas. En una versión avanzada se puede sustituir por embeddings y un modelo generativo, conservando la misma interfaz.