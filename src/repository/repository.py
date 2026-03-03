"""
Módulo de gerenciamento de dados de reconhecimento facial.

Este módulo é responsável pelo armazenamento, processamento e recuperação
dos encodings faciais utilizados pelo sistema de reconhecimento.

Principais responsabilidades:
- Carregar e salvar encodings faciais em cache utilizando arquivos pickle.
- Extrair encodings a partir de imagens ou vídeos de alunos.
- Gerenciar o cache de encodings para evitar recomputações custosas.
- Permitir a adição incremental de novos rostos ao sistema.
"""

from pathlib import Path
import shutil
import pickle
import tkinter as tk
from tkinter import filedialog
import face_recognition
import cv2
import numpy as np

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances

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

def import_student_video(student_name: str) -> bool:
    """
    Permite selecionar um vídeo de um aluno e copiá-lo para o diretório
    de dados utilizado para extração de encodings faciais.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    source_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")]
    )

    if not source_path:
        return False

    dest_dir = BASE_DATA_DIR / "raw_face_video"
    dest_dir.mkdir(parents=True, exist_ok=True)

    extension = Path(source_path).suffix
    dest_path = dest_dir / f"{student_name}{extension}"

    shutil.copy2(source_path, dest_path)
    return True

def extract_encodings_from_selfie_video(file_path, sample_every=8, scale=0.5, model="hog"):
    """
    Extrai encodings faciais de um vídeo de selfie amostrando frames
    periodicamente e filtrando rostos com baixa qualidade.
    """
    cap = cv2.VideoCapture(str(file_path))
    person_encodings = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        if frame_count % sample_every != 0:
            continue

        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        faces = face_recognition.face_locations(rgb, model=model)
        if not faces:
            continue

        # pega o rosto de maior area
        areas = [(b - t) * (r - l) for (t, r, b, l) in faces]
        idx = int(np.argmax(areas))
        t, r, b, l = faces[idx]

        crop = small[t:b, l:r]
        if crop.size == 0:
            continue
        #filtra rostos borrados
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        MIN_AREA = 5000
        MIN_BLUR = 40

        if areas[idx] < MIN_AREA or lap_var < MIN_BLUR:
            continue

        encoding = face_recognition.face_encodings(rgb, [(t, r, b, l)], model="large")
        if encoding:
            person_encodings.append(encoding[0])

    cap.release()
    return person_encodings

def calculate_know_face_video_encodings(save_cache=True):
    '''
    Calcula os face encodings de uma pessoa baseado no video de apresentação.
    Salva 3 encondings para cada pessoa.
    '''
    print("--- INICIANDO PROCESSAMENTO DE VIDEOS---")
    known_face_encodings = []
    known_face_names = []

    raw_path = BASE_DATA_DIR / "raw_face_video"

    if not raw_path.exists():
        print(f"Pasta não encontrada: {raw_path}")
        return [], []

    for file_path in raw_path.iterdir():
        if file_path.suffix.lower() not in ['.mp4', '.avi', '.mov']:
            continue

        name = file_path.stem
        print(f"Extraindo características de: {name}...", end=" ")

        #faz o calculo do encoding
        person_encodings = extract_encodings_from_selfie_video(file_path, scale=1)

        num_frames = len(person_encodings)
        if num_frames > 0:
            encs = np.array(person_encodings)
            # Tentar remover ruído com DBSCAN se falhar voltar para KMeans simples.
            if len(encs) >= 3:
                db = DBSCAN(eps=0.55, min_samples=2, metric='euclidean').fit(encs)
                labels = db.labels_
                unique_labels = [lab for lab in set(labels) if lab != -1]
                if len(unique_labels) == 0:
                    print("Tudo ruido: FALLBACK PRIMARIO")
                    # tudo foi marcado como ruído; fallback para KMeans com k=1
                    kmeans = KMeans(n_clusters=1, n_init="auto", random_state=42).fit(encs)
                    for centroid in kmeans.cluster_centers_:
                        known_face_encodings.append(centroid)
                        known_face_names.append(name)
                else:
                    # para cada cluster, escolher o medoid
                    for lab in unique_labels:
                        cluster = encs[labels == lab]
                        D = pairwise_distances(cluster)
                        medoid = cluster[np.argmin(D.sum(axis=1))]
                        known_face_encodings.append(medoid)
                        known_face_names.append(name)
            else:
                print("Poucas Amostras: FALLBACK SECUNDARIO")
                # se poucas amostras, use KMeans com K = num amostras (ou 1)
                k_clusters = min(3, len(encs))
                kmeans = KMeans(n_clusters=k_clusters, n_init="auto", random_state=42).fit(encs)
                for centroid in kmeans.cluster_centers_:
                    known_face_encodings.append(centroid)
                    known_face_names.append(name)
        else:
            print("FALHA: Nenhum rosto nítido detectado no vídeo inteiro.")

    if save_cache:
        _save_pickle(known_face_encodings, KNOW_FACE_ENCODINGS_FILE_NAME)
        _save_pickle(known_face_names, KNOW_FACE_NAMES_FILE_NAME)
        print("Cache de vídeos salvo com sucesso!")

    return known_face_encodings, known_face_names

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
        return calculate_know_face_video_encodings()

    know_face_encodings = _load_pickle(KNOW_FACE_ENCODINGS_FILE_NAME)
    know_face_names = _load_pickle(KNOW_FACE_NAMES_FILE_NAME)

    if know_face_encodings is None or know_face_names is None:
        print("Cache não encontrado, recalculando")
        return _calculate_know_face_encodings(save_cache)

    print("Usando cache")
    return know_face_encodings, know_face_names


def rebuild_cache():
    """Força o rebuild total do cache."""
    calculate_know_face_video_encodings(save_cache=True)


def add_single_face(image_path, name_override=None):
    """
    Adiciona uma única pessoa ao cache existente sem recalcular tudo.
    Útil para cadastrar UM novo aluno
    """
    encodings, names = get_know_face_encodings(recalculate=False)

    img = cv2.imread(str(image_path))
    if img is None:
        return False

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
    print("Nenhum rosto encontrado")
    return False
