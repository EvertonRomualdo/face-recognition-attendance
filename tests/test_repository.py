import numpy as np
from repository import repository

def test_load_pickle_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(repository, "PICKLE_DIR", tmp_path)
    assert repository._load_pickle("nope.pkl") is None

def test_save_and_load_pickle_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(repository, "PICKLE_DIR", tmp_path)
    data = [np.zeros(3), np.ones(3)]
    repository._save_pickle(data, "x.pkl")
    loaded = repository._load_pickle("x.pkl")
    assert len(loaded) == 2
    assert (loaded[0] == data[0]).all()

def test_get_known_face_encodings_uses_cache_when_available(mocker):
    fake_enc = [np.zeros(128)]
    fake_names = ["Anderson"]
    mocker.patch("repository.repository._load_pickle", side_effect=[fake_enc, fake_names])

    enc, names = repository.get_know_face_encodings(recalculate=False)
    assert enc == fake_enc
    assert names == fake_names

def test_get_known_face_encodings_recalculates_when_cache_missing(mocker):
    load = mocker.patch("repository.repository._load_pickle", side_effect=[None, None])
    calc = mocker.patch("repository.repository._calculate_know_face_encodings", return_value=(["e"], ["n"]))

    enc, names = repository.get_know_face_encodings()
    assert enc == ["e"]
    assert names == ["n"]
    assert load.call_count == 2
    calc.assert_called_once()

def test_get_known_face_encodings_forces_recalculate(mocker):
    mocker.patch(
        "repository.repository._calculate_know_face_encodings",
        return_value=(["enc"], ["name"])
    )

    enc, names = repository.get_know_face_encodings(recalculate=True)

    assert enc == ["enc"]
    assert names == ["name"]


def test_add_single_face_returns_false_when_imread_fails(mocker):
    mocker.patch("cv2.imread", return_value=None)
    assert repository.add_single_face("fake.jpg") is False

def test_add_single_face_adds_and_persists_when_face_found(mocker, tmp_path):
    mocker.patch.object(repository, "PICKLE_DIR", tmp_path)
    mocker.patch("repository.repository._load_pickle", side_effect=[[], []])
    mocker.patch("cv2.imread", return_value=np.zeros((100,100,3), dtype=np.uint8))
    mocker.patch("cv2.cvtColor", side_effect=lambda img, code: img)
    mocker.patch("face_recognition.face_encodings", return_value=[np.ones(128)])

    ok = repository.add_single_face("aluno.png", name_override="Anderson")
    assert ok is True

    assert (tmp_path / repository.KNOW_FACE_ENCODINGS_FILE_NAME).exists()
    assert (tmp_path / repository.KNOW_FACE_NAMES_FILE_NAME).exists()

def test_add_single_face_returns_false_when_no_face_found(mocker):
    mocker.patch("repository.repository.get_know_face_encodings", return_value=([], []))

    fake_img = np.zeros((10, 10, 3), dtype=np.uint8)
    mocker.patch("cv2.imread", return_value=fake_img)
    mocker.patch("cv2.cvtColor", return_value=fake_img)
    mocker.patch("face_recognition.face_encodings", return_value=[])

    assert repository.add_single_face("x.jpg") is False

def test_rebuild_cache_calls_calculate(mocker):
    mock_calc = mocker.patch(
        "repository.repository._calculate_know_face_encodings"
    )

    from repository.repository import rebuild_cache
    rebuild_cache()

    mock_calc.assert_called_once_with(save_cache=True)

def test_calculate_known_face_encodings_ignores_non_image_files(mocker, tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw_images"
    raw_dir.mkdir()

    (raw_dir / "arquivo.txt").write_text("x")
    (raw_dir / "video.mp4").write_text("y")

    monkeypatch.setattr(repository, "BASE_DATA_DIR", tmp_path)
    mock_imread = mocker.patch("cv2.imread")
    enc, names = repository._calculate_know_face_encodings(save_cache=False)

    assert enc == []
    assert names == []
    mock_imread.assert_not_called()

def test_calculate_known_face_encodings_skips_when_imread_returns_none(mocker, tmp_path):
    mocker.patch("repository.repository.BASE_DATA_DIR", tmp_path)
    raw = tmp_path / "raw_images"
    raw.mkdir()
    (raw / "anderson.jpg").write_bytes(b"x")

    mocker.patch("cv2.imread", return_value=None)

    enc, names = repository._calculate_know_face_encodings(save_cache=False)
    assert enc == []
    assert names == []

def test_calculate_known_face_encodings_handles_no_faces(mocker, tmp_path):
    mocker.patch("repository.repository.BASE_DATA_DIR", tmp_path)
    raw = tmp_path / "raw_images"
    raw.mkdir()
    (raw / "anderson.jpg").write_bytes(b"x")

    fake_img = np.zeros((10, 10, 3), dtype=np.uint8)
    mocker.patch("cv2.imread", return_value=fake_img)
    mocker.patch("cv2.cvtColor", return_value=fake_img)
    mocker.patch("face_recognition.face_encodings", return_value=[])

    enc, names = repository._calculate_know_face_encodings(save_cache=False)
    assert enc == []
    assert names == []

def test_calculate_known_face_encodings_returns_empty_when_raw_dir_missing(tmp_path, mocker):
    mocker.patch("repository.repository.BASE_DATA_DIR", tmp_path)
    enc, names = repository._calculate_know_face_encodings(save_cache=False)
    assert enc == []
    assert names == []

def test_calculate_known_face_encodings_appends_when_face_found(mocker, tmp_path):
    mocker.patch("repository.repository.BASE_DATA_DIR", tmp_path)
    raw = tmp_path / "raw_images"
    raw.mkdir()
    (raw / "anderson.jpg").write_bytes(b"x")

    fake_img = np.zeros((10, 10, 3), dtype=np.uint8)
    mocker.patch("cv2.imread", return_value=fake_img)
    mocker.patch("cv2.cvtColor", return_value=fake_img)
    mocker.patch("face_recognition.face_encodings", return_value=[np.ones(128)])

    enc, names = repository._calculate_know_face_encodings(save_cache=False)

    assert len(enc) == 1
    assert len(names) == 1
    assert names[0] == "anderson"


