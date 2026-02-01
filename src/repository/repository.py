import pickle
import face_recognition
import cv2
import numpy as np
from pathlib import Path

BASE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PICKLE_DIR = BASE_DATA_DIR / "pickle_cache"
PICKLE_DIR.mkdir(parents=True, exist_ok=True)

KNOW_FACE_ENCODINGS_FILE_NAME = "know_face_encodings.pkl"
KNOW_FACE_NAMES_FILE_NAME = "know_face_names.pkl"


def _save_pickle(data, filename):
    filepath = PICKLE_DIR / filename
    with open(filepath, "wb") as file:
        pickle.dump(data, file)


def _load_pickle(filename):
    filepath = PICKLE_DIR / filename
    if not filepath.exists():
        return None
    with open(filepath, "rb") as file:
        return pickle.load(file)


def _calculate_know_face_encodings(save_cache=True):
    known_face_encodings = []
    known_face_names = []

    raw_path = BASE_DATA_DIR / "raw_images"

    if not raw_path.exists():
        return [], []

    for file_path in raw_path.iterdir():
        if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        name = file_path.stem

        try:
            img = cv2.imread(str(file_path))
            if img is None:
                print("ERRO LEITURA")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.ascontiguousarray(img)

            encodings = face_recognition.face_encodings(img)

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
            else:
                print("Sem rosto detectado")

        except Exception as e:
            print(f"Erro ao processar {name}: {e}")

    if save_cache:
        _save_pickle(known_face_encodings, KNOW_FACE_ENCODINGS_FILE_NAME)
        _save_pickle(known_face_names, KNOW_FACE_NAMES_FILE_NAME)
        print("Cache salvo")

    return known_face_encodings, known_face_names


def get_know_face_encodings(save_cache=True, recalculate=False):
    """Retorna uma lista com os encodings e outra com os nomes em ordem"""

    if recalculate:
        print("Forçando recálculo")
        return _calculate_know_face_encodings(save_cache)

    know_face_encodings = _load_pickle(KNOW_FACE_ENCODINGS_FILE_NAME)
    know_face_names = _load_pickle(KNOW_FACE_NAMES_FILE_NAME)

    if know_face_encodings is None or know_face_names is None:
        print("Cache não encontrado, recalculando")
        return _calculate_know_face_encodings(save_cache)

    print("Usando cache")
    return know_face_encodings, know_face_names


def rebuild_cache():
    """Força o rebuild total do cache."""
    _calculate_know_face_encodings(save_cache=True)


def add_single_face(image_path, name_override=None):
    """
    Adiciona uma única pessoa ao cache existente sem recalcular tudo.
    Útil para cadastrar UM novo aluno
    """
    encodings, names = get_know_face_encodings(recalculate=False)

    img = cv2.imread(str(image_path))
    if img is None: return False

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img)

    new_encs = face_recognition.face_encodings(img)

    if len(new_encs) > 0:
        new_name = name_override if name_override else Path(image_path).stem
        encodings.append(new_encs[0])
        names.append(new_name)

        _save_pickle(encodings, KNOW_FACE_ENCODINGS_FILE_NAME)
        _save_pickle(names, KNOW_FACE_NAMES_FILE_NAME)
        print("adicionado")
        return True
    else:
        print("Nenhum rosto encontrado")
        return False











