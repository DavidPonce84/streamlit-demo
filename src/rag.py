"""Offline lexical RAG engine for a reproducible classroom demo."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    source: str
    text: str


def load_documents(directory: str | Path) -> list[Document]:
    """Load Markdown/text documents from a directory in sorted order."""
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"No existe el directorio de documentos: {path}")
    docs = []
    for file in sorted(path.glob("*")):
        if file.suffix.lower() in {".md", ".txt"}:
            docs.append(Document(file.name, file.read_text(encoding="utf-8")))
    if not docs:
        raise ValueError("No se encontraron documentos .md o .txt")
    return docs


class LocalRAG:
    """Simple term-overlap retrieval with cited excerpts, no API required."""

    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("Se requiere al menos un documento")
        self.documents = documents

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {word.lower() for word in re.findall(r"[\wáéíóúñü]+", text) if len(word) > 2}

    def retrieve(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        if not query or not query.strip():
            raise ValueError("La pregunta no puede estar vacía")
        query_terms = self._terms(query)
        scored = []
        for doc in self.documents:
            doc_terms = self._terms(doc.text)
            score = len(query_terms & doc_terms) / max(len(query_terms), 1)
            scored.append((doc, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:k]

    def answer(self, query: str, k: int = 2) -> dict:
        results = self.retrieve(query, k=k)
        useful = [(doc, score) for doc, score in results if score > 0]
        if not useful:
            return {"answer": "No encontré evidencia suficiente en los documentos cargados.", "sources": [], "grounded": False}
        excerpts = []
        for doc, score in useful:
            lines = [line.strip() for line in doc.text.splitlines() if line.strip()]
            excerpts.append({"source": doc.source, "score": round(score, 3), "excerpt": " ".join(lines[:3])[:500]})
        answer = "Con base en la documentación disponible:\n\n" + "\n\n".join(f"- {item['excerpt']}" for item in excerpts)
        return {"answer": answer, "sources": excerpts, "grounded": True}
