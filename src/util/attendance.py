"""
Módulo de persistência de registros de presença.

Este módulo implementa funcionalidades responsáveis por registrar e armazenar
a presença dos alunos reconhecidos pelo sistema de reconhecimento facial.

Principais funcionalidades:
- Gerar nomes de arquivos de presença baseados em timestamp ou nome de sessão.
- Salvar listas de alunos reconhecidos em arquivos CSV.
- Adicionar registros de presença em tempo real (append).
"""

from datetime import datetime
from pathlib import Path
import csv

BASE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
ATTENDANCE_DIR = BASE_DATA_DIR / "attendance"
ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)

def get_attendance_filename(session_name=None):
    """
    Gera o nome do arquivo de presença baseado na data/hora.
    Se session_name for fornecido, usa esse nome, caso contrário usa timestamp.
    """
    if session_name:
        return ATTENDANCE_DIR / f"{session_name}.csv"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ATTENDANCE_DIR / f"attendance_{timestamp}.csv"

def save_attendance(recognized_students, session_name=None):
    """
    Salva a lista de alunos reconhecidos em um arquivo CSV.

    Args:
        recognized_students: Dict com {nome: datetime_str} dos alunos reconhecidos
        session_name: Nome opcional da sessão (se None, usa timestamp)

    Returns:
        Path do arquivo criado
    """
    filepath = get_attendance_filename(session_name)

    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Nome', 'Data/Hora Reconhecimento']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # Ordena por timestamp
        sorted_students = sorted(recognized_students.items(),
                                key=lambda x: x[1])

        for name, timestamp_str in sorted_students:
            writer.writerow({
                'Nome': name,
                'Data/Hora Reconhecimento': timestamp_str
            })

    print(f"✅ Arquivo de presença salvo: {filepath}")
    return filepath


def append_attendance(name, filepath=None):
    """
    Adiciona um aluno ao arquivo de presença (útil para logging em tempo real).
    Se o arquivo não existir, cria um novo.

    Args:
        name: Nome do aluno
        filepath: Caminho do arquivo (se None, usa o padrão do dia)

    Returns:
        Path do arquivo atualizado
    """
    if filepath is None:
        # Usa o arquivo de hoje
        today = datetime.now().strftime("%Y%m%d")
        filepath = ATTENDANCE_DIR / f"attendance_{today}.csv"

    filepath = Path(filepath)
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Verifica se o arquivo existe
    file_exists = filepath.exists()

    with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Nome', 'Data/Hora Reconhecimento']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Escreve header se o arquivo é novo
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'Nome': name,
            'Data/Hora Reconhecimento': timestamp_str
        })

    return filepath
