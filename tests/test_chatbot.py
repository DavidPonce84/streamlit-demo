from src.chatbot import answer_controlled, chatbot_status
from src.rag import Document, LocalRAG


def test_chatbot_status_does_not_expose_secret():
    status = chatbot_status()
    assert "api_key" not in status
    assert isinstance(status["configured"], bool)


def test_controlled_chatbot_is_grounded_and_local():
    rag = LocalRAG([Document("guide.md", "Streamlit se ejecuta con streamlit run app.py.")])
    result = answer_controlled(rag, "¿Cómo se ejecuta Streamlit?")
    assert result["mode"] == "local-grounded"
    assert result["provider"] == "none"
    assert result["grounded"] is True


def test_controlled_chatbot_rejects_unsupported_question():
    rag = LocalRAG([Document("guide.md", "Streamlit es una herramienta web.")])
    result = answer_controlled(rag, "¿Cuál es mi contraseña?")
    assert result["grounded"] is False
    assert result["sources"] == []
