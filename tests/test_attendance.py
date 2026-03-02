import csv
from datetime import datetime as real_datetime

import pytest

from etl import attendance

class FixedDatetime:
    """
    Substitui datetime no módulo para deixar os testes determinísticos.
    """
    @classmethod
    def now(cls):
        # 2026-02-10 13:05:09
        return real_datetime(2026, 2, 10, 13, 5, 9)

@pytest.fixture()
def isolated_attendance_dir(tmp_path, monkeypatch):
    base = tmp_path / "data"
    att_dir = base / "attendance"
    att_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(attendance, "BASE_DATA_DIR", base)
    monkeypatch.setattr(attendance, "ATTENDANCE_DIR", att_dir)
    monkeypatch.setattr(attendance, "datetime", FixedDatetime)

    return att_dir

def read_csv_rows(filepath):
    with open(filepath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def test_get_attendance_filename_with_session_name(isolated_attendance_dir):
    p = attendance.get_attendance_filename(session_name="aula_IA")
    assert p.parent == isolated_attendance_dir
    assert p.name == "aula_IA.csv"

def test_get_attendance_filename_without_session_uses_timestamp(isolated_attendance_dir):
    p = attendance.get_attendance_filename(session_name=None)
    assert p.parent == isolated_attendance_dir
    assert p.name == "attendance_20260210_130509.csv"

def test_save_attendance_creates_csv_sorted_by_timestamp(isolated_attendance_dir, capsys):
    recognized = {
        "Arlis": "2026-02-10 13:05:08",
        "Anderson": "2026-02-10 13:05:01",
        "Everton": "2026-02-10 13:05:05",
    }

    out_path = attendance.save_attendance(recognized, session_name="sessao1")
    assert out_path.exists()
    assert out_path.name == "sessao1.csv"

    rows = read_csv_rows(out_path)
    assert [r["Nome"] for r in rows] == ["Anderson", "Everton", "Arlis"]

    captured = capsys.readouterr()
    assert "Arquivo de presença salvo" in captured.out

def test_append_attendance_creates_file_and_writes_header_once(isolated_attendance_dir):
    out_path = attendance.append_attendance("Anderson", filepath=None)
    assert out_path.exists()
    assert out_path.name == "attendance_20260210.csv"

    rows = read_csv_rows(out_path)
    assert len(rows) == 1
    assert rows[0]["Nome"] == "Anderson"
    assert rows[0]["Data/Hora Reconhecimento"] == "2026-02-10 13:05:09"

def test_append_attendance_appends_without_duplicate_header(isolated_attendance_dir):
    filepath = isolated_attendance_dir / "minha_sessao.csv"

    attendance.append_attendance("A", filepath=filepath)
    attendance.append_attendance("B", filepath=filepath)

    text = filepath.read_text(encoding="utf-8").splitlines()
    header_count = sum(1 for line in text if line.strip() == "Nome,Data/Hora Reconhecimento")
    assert header_count == 1

    rows = read_csv_rows(filepath)
    assert [r["Nome"] for r in rows] == ["A", "B"]

def test_append_attendance_accepts_string_path(isolated_attendance_dir):
    filepath_str = str(isolated_attendance_dir / "strpath.csv")
    out_path = attendance.append_attendance("Anderson", filepath=filepath_str)
    assert out_path.exists()
    assert out_path.name == "strpath.csv"