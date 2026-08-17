"""Educational Streamlit AI showcase components."""
__version__ = "1.0.0"

from .tabular import build_demo_dataset, train_tabular_model, predict_tabular
from .explainability import global_importance, local_explanation, lime_available
from .rag import LocalRAG, load_documents
from .vision import classify_image

__all__ = [
    "build_demo_dataset", "train_tabular_model", "predict_tabular",
    "global_importance", "local_explanation", "lime_available",
    "LocalRAG", "load_documents", "classify_image",
]
