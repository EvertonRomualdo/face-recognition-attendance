import numpy as np
from repository import repository
import pytest

@pytest.fixture(autouse=True)
def block_video_processing(request, mocker):
    if request.node.get_closest_marker("allow_video"):
        return

    mocker.patch(
        "repository.repository.extract_encodings_from_selfie_video",
        side_effect=AssertionError(
            "Teste tentou processar vídeo real. Use @pytest.mark.allow_video e mocke o pipeline."
        ),
    )

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
    mock_video_calc = mocker.patch(
        "repository.repository.calculate_know_face_video_encodings",
        return_value=(["enc"], ["name"])
    )

    enc, names = repository.get_know_face_encodings(recalculate=True)

    assert enc == ["enc"]
    assert names == ["name"]
    mock_video_calc.assert_called_once()


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
    mock_calc = mocker.patch("repository.repository.calculate_know_face_video_encodings")
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

def test_import_student_video_returns_false_when_user_cancels(mocker, tmp_path):
    mocker.patch("repository.repository.BASE_DATA_DIR", tmp_path)

    fake_root = mocker.Mock()
    fake_root.withdraw = mocker.Mock()
    fake_root.attributes = mocker.Mock()
    mocker.patch("tkinter.Tk", return_value=fake_root)

    ask = mocker.patch("tkinter.filedialog.askopenfilename", return_value="")
    copy2 = mocker.patch("shutil.copy2")

    ok = repository.import_student_video("Anderson")
    assert ok is False
    ask.assert_called_once()
    copy2.assert_not_called()


def test_import_student_video_copies_file_when_selected(mocker, tmp_path):
    mocker.patch("repository.repository.BASE_DATA_DIR", tmp_path)

    fake_root = mocker.Mock()
    fake_root.withdraw = mocker.Mock()
    fake_root.attributes = mocker.Mock()
    mocker.patch("tkinter.Tk", return_value=fake_root)

    source = "/tmp/any_video.mp4"
    mocker.patch("tkinter.filedialog.askopenfilename", return_value=source)

    copy2 = mocker.patch("shutil.copy2")

    ok = repository.import_student_video("Anderson")
    assert ok is True

    dest_dir = tmp_path / "raw_face_video"
    assert dest_dir.exists()

    dest_path = dest_dir / "Anderson.mp4"
    copy2.assert_called_once_with(source, dest_path)


# ---------------------------------------------------------
# NOVOS TESTES: extract_encodings_from_selfie_video
# ---------------------------------------------------------
@pytest.mark.allow_video
def test_extract_encodings_returns_one_encoding_when_face_is_good(mocker):
    # Mock do VideoCapture (3 frames: só o 2º processa com sample_every=2)
    cap = mocker.Mock()
    cap.read.side_effect = [
        (True, np.zeros((200, 200, 3), dtype=np.uint8)),
        (True, np.zeros((200, 200, 3), dtype=np.uint8)),
        (False, None),
    ]
    cap.release = mocker.Mock()
    mocker.patch("cv2.VideoCapture", return_value=cap)

    # Simplifica operações de imagem
    mocker.patch("cv2.resize", side_effect=lambda frame, size, fx, fy: frame)

    def fake_cvt(img, code):
        # RGB -> mantém shape 3 canais; GRAY -> 2D
        if len(img.shape) == 3:
            return img
        return img

    mocker.patch("cv2.cvtColor", side_effect=fake_cvt)

    # Face grande e nítida
    mocker.patch("face_recognition.face_locations", return_value=[(0, 100, 100, 0)])
    enc_mock = mocker.patch("face_recognition.face_encodings", return_value=[np.ones(128)])

    # blur ok: var > 40
    mocker.patch("cv2.Laplacian", return_value=np.array([0.0, 100.0]))

    out = repository.extract_encodings_from_selfie_video("x.mp4", sample_every=2, scale=1)

    assert len(out) == 1
    assert out[0].shape == (128,)
    enc_mock.assert_called_once()
    cap.release.assert_called_once()


@pytest.mark.allow_video
def test_extract_encodings_filters_blurry_face_and_returns_empty(mocker):
    cap = mocker.Mock()
    cap.read.side_effect = [
        (True, np.zeros((200, 200, 3), dtype=np.uint8)),
        (False, None),
    ]
    cap.release = mocker.Mock()
    mocker.patch("cv2.VideoCapture", return_value=cap)

    mocker.patch("cv2.resize", side_effect=lambda frame, size, fx, fy: frame)
    mocker.patch("cv2.cvtColor", side_effect=lambda img, code: img)

    # Face grande, mas blur ruim (var ~ 0)
    mocker.patch("face_recognition.face_locations", return_value=[(0, 100, 100, 0)])
    enc_mock = mocker.patch("face_recognition.face_encodings", return_value=[np.ones(128)])
    mocker.patch("cv2.Laplacian", return_value=np.array([0.0]))  # var = 0 -> blur ruim

    out = repository.extract_encodings_from_selfie_video("x.mp4", sample_every=1, scale=1)

    assert out == []
    enc_mock.assert_not_called()
    cap.release.assert_called_once()


# ---------------------------------------------------------
# NOVOS TESTES: calculate_know_face_video_encodings (branches DBSCAN/KMeans)
# ---------------------------------------------------------
@pytest.mark.allow_video
def test_calculate_video_encodings_returns_empty_when_video_dir_missing(mocker, tmp_path):
    mocker.patch("repository.repository.BASE_DATA_DIR", tmp_path)

    enc, names = repository.calculate_know_face_video_encodings(save_cache=False)
    assert enc == []
    assert names == []


@pytest.mark.allow_video
def test_calculate_video_encodings_ignores_non_video_files(mocker, tmp_path):
    mocker.patch("repository.repository.BASE_DATA_DIR", tmp_path)
    raw = tmp_path / "raw_face_video"
    raw.mkdir()
    (raw / "x.txt").write_text("nope")

    ext = mocker.patch("repository.repository.extract_encodings_from_selfie_video")

    enc, names = repository.calculate_know_face_video_encodings(save_cache=False)
    assert enc == []
    assert names == []
    ext.assert_not_called()


@pytest.mark.allow_video
def test_calculate_video_encodings_fallback_secondary_kmeans_when_few_samples(mocker, tmp_path):
    mocker.patch("repository.repository.BASE_DATA_DIR", tmp_path)
    raw = tmp_path / "raw_face_video"
    raw.mkdir()
    (raw / "Anderson.mp4").write_bytes(b"x")

    # 2 amostras -> fallback secundário (KMeans com k=2)
    mocker.patch(
        "repository.repository.extract_encodings_from_selfie_video",
        return_value=[np.zeros(128), np.ones(128)],
    )

    class FakeKMeans:
        def __init__(self, n_clusters, n_init, random_state):
            self.n_clusters = n_clusters
            self.cluster_centers_ = [np.full(128, 10.0), np.full(128, 20.0)]

        def fit(self, encs):
            return self

    km = mocker.patch("repository.repository.KMeans", side_effect=FakeKMeans)

    enc, names = repository.calculate_know_face_video_encodings(save_cache=False)

    assert len(enc) == 2
    assert names == ["Anderson", "Anderson"]
    assert km.called


@pytest.mark.allow_video
def test_calculate_video_encodings_fallback_primary_when_dbscan_all_noise(mocker, tmp_path):
    mocker.patch("repository.repository.BASE_DATA_DIR", tmp_path)
    raw = tmp_path / "raw_face_video"
    raw.mkdir()
    (raw / "Anderson.mp4").write_bytes(b"x")

    # 3+ amostras -> tenta DBSCAN
    mocker.patch(
        "repository.repository.extract_encodings_from_selfie_video",
        return_value=[np.zeros(128), np.ones(128), np.full(128, 2.0)],
    )

    # DBSCAN: tudo -1 (ruído)
    class FakeDBSCAN:
        def __init__(self, eps, min_samples, metric):
            self.labels_ = None

        def fit(self, encs):
            self.labels_ = np.array([-1, -1, -1])
            return self

    mocker.patch("repository.repository.DBSCAN", side_effect=FakeDBSCAN)

    # fallback primário: KMeans com k=1
    class FakeKMeans:
        def __init__(self, n_clusters, n_init, random_state):
            self.cluster_centers_ = [np.full(128, 99.0)]

        def fit(self, encs):
            return self

    mocker.patch("repository.repository.KMeans", side_effect=FakeKMeans)

    enc, names = repository.calculate_know_face_video_encodings(save_cache=False)

    assert len(enc) == 1
    assert names == ["Anderson"]


@pytest.mark.allow_video
def test_calculate_video_encodings_dbscan_clusters_choose_medoid(mocker, tmp_path):
    mocker.patch("repository.repository.BASE_DATA_DIR", tmp_path)
    raw = tmp_path / "raw_face_video"
    raw.mkdir()
    (raw / "Anderson.mp4").write_bytes(b"x")

    # 4 encodings -> DBSCAN labels [0,0,1,1]
    e0 = np.zeros(128)
    e1 = np.ones(128)
    e2 = np.full(128, 2.0)
    e3 = np.full(128, 3.0)

    mocker.patch(
        "repository.repository.extract_encodings_from_selfie_video",
        return_value=[e0, e1, e2, e3],
    )

    class FakeDBSCAN:
        def __init__(self, eps, min_samples, metric):
            self.labels_ = None

        def fit(self, encs):
            self.labels_ = np.array([0, 0, 1, 1])
            return self

    mocker.patch("repository.repository.DBSCAN", side_effect=FakeDBSCAN)

    # pairwise_distances: tudo zero -> medoid vira o primeiro item do cluster
    mocker.patch("repository.repository.pairwise_distances", side_effect=lambda cluster: np.zeros((len(cluster), len(cluster))))

    enc, names = repository.calculate_know_face_video_encodings(save_cache=False)

    # um medoid por cluster: e0 (cluster 0) e e2 (cluster 1)
    assert len(enc) == 2
    assert names == ["Anderson", "Anderson"]
    assert np.allclose(enc[0], e0)
    assert np.allclose(enc[1], e2)
