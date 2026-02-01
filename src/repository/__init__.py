from .repository import get_know_face_encodings, rebuild_cache, add_single_face

# Isso define o que é exportado quando alguém faz "from repository import *"
# E serve como documentação do que é público.
__all__ = [
    "get_know_face_encodings",
    "rebuild_cache",
    "add_single_face"
]