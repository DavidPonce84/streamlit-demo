from pathlib import Path

import pytest

from src.rag import Document, LocalRAG, load_documents


def test_documents_load_from_repository():
    docs = load_documents(Path(__file__).parents[1] / "documents")
    assert len(docs) >= 3
    assert all(doc.text for doc in docs)


def test_rag_retrieves_grounded_answer():
    rag = LocalRAG([Document("guide.md", "Streamlit ejecuta aplicaciones con streamlit run app.py.")])
    result = rag.answer("¿Cómo se ejecuta Streamlit?")
    assert result["grounded"] is True
    assert result["sources"][0]["source"] == "guide.md"
    assert "streamlit run" in result["answer"]


def test_rag_declares_missing_evidence():
    rag = LocalRAG([Document("guide.md", "Streamlit es una herramienta web.")])
    result = rag.answer("¿Cuál es la política de vacaciones?")
    assert result["grounded"] is False
    assert result["sources"] == []


def test_rag_rejects_empty_query():
    rag = LocalRAG([Document("guide.md", "Contenido")])
    with pytest.raises(ValueError):
        rag.retrieve("  ")
