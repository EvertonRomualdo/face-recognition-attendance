import numpy as np
import pytest

from app import application

def _fake_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_get_video_capture_returns_none_when_camera_fails(mocker):
    cap_mock = mocker.Mock()
    cap_mock.isOpened.return_value = False
    mocker.patch("cv2.VideoCapture", return_value=cap_mock)

    result = application.get_video_capture(0)
    assert result is None


def test_execute_recognization_registers_presence_at_interval(mocker):
    mocker.patch(
        "app.application.repository.get_know_face_encodings",
        return_value=([np.zeros(128)], ["Anderson"])
    )

    cap = mocker.Mock()
    cap.read.side_effect = [
        (True, _fake_frame()),
        (True, _fake_frame()),
        (False, None),
    ]

    mocker.patch("cv2.resize", side_effect=lambda frame, *_args, **_kw: frame)

    mocker.patch("face_recognition.face_locations", return_value=[(1, 2, 3, 4)])
    mocker.patch("face_recognition.face_encodings", return_value=[np.zeros(128)])
    mocker.patch("face_recognition.compare_faces", return_value=[True])
    mocker.patch("face_recognition.face_distance", return_value=[0.0])

    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=0)  # não aperta q
    mocker.patch("cv2.destroyAllWindows")

    application.execute_recognization(cap, process_interval=2, scale_factor=0.5)

    cap.release.assert_called_once()

def test_execute_recognization_calls_face_pipeline_only_every_n_frames(mocker):
    mocker.patch(
        "app.application.repository.get_know_face_encodings",
        return_value=([np.zeros(128)], ["Anderson"])
    )

    cap = mocker.Mock()
    cap.read.side_effect = [
        (True, _fake_frame()),
        (True, _fake_frame()),
        (True, _fake_frame()), 
        (True, _fake_frame()),
        (True, _fake_frame()),
        (True, _fake_frame()),
        (False, None),
    ]

    mocker.patch("cv2.resize", side_effect=lambda frame, *_args, **_kw: frame)

    loc_mock = mocker.patch("face_recognition.face_locations", return_value=[(1, 2, 3, 4)])
    enc_mock = mocker.patch("face_recognition.face_encodings", return_value=[np.zeros(128)])
    mocker.patch("face_recognition.compare_faces", return_value=[True])
    mocker.patch("face_recognition.face_distance", return_value=[0.0])

    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=0)
    mocker.patch("cv2.destroyAllWindows")

    application.execute_recognization(cap, process_interval=3, scale_factor=0.5)

    assert loc_mock.call_count == 2
    assert enc_mock.call_count == 2


@pytest.mark.parametrize(
    "detected_name, expected_color",
    [
        ("Desconhecido", (0, 0, 255)),
        ("Anderson", (0, 255, 0)),
    ],
)
def test_execute_recognization_draws_correct_color(mocker, detected_name, expected_color):
    mocker.patch(
        "app.application.repository.get_know_face_encodings",
        return_value=([np.zeros(128)], ["Anderson"])
    )

    cap = mocker.Mock()
    cap.read.side_effect = [
        (True, _fake_frame()),
        (True, _fake_frame()),
        (False, None),
    ]

    mocker.patch("cv2.resize", side_effect=lambda frame, *_args, **_kw: frame)
    mocker.patch("face_recognition.face_locations", return_value=[(10, 20, 30, 40)])
    mocker.patch("face_recognition.face_encodings", return_value=[np.zeros(128)])

    if detected_name == "Anderson":
        mocker.patch("face_recognition.compare_faces", return_value=[True])
        mocker.patch("face_recognition.face_distance", return_value=[0.0])
    else:
        mocker.patch("face_recognition.compare_faces", return_value=[False])
        mocker.patch("face_recognition.face_distance", return_value=[1.0])

    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=0)
    mocker.patch("cv2.destroyAllWindows")

    rect_mock = mocker.patch("cv2.rectangle")
    application.execute_recognization(cap, process_interval=2, scale_factor=0.5)

    assert rect_mock.call_args_list, "cv2.rectangle não foi chamado"
    _, args, _ = rect_mock.mock_calls[0]
    assert args[3] == expected_color

def test_execute_recognization_uses_scale_factor_in_resize(mocker):
    mocker.patch(
        "app.application.repository.get_know_face_encodings",
        return_value=([], [])
    )

    cap = mocker.Mock()
    cap.read.side_effect = [
        (True, _fake_frame()),
        (False, None),
    ]

    resize_mock = mocker.patch("cv2.resize", side_effect=lambda frame, *_args, **_kw: frame)
    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=0)
    mocker.patch("cv2.destroyAllWindows")

    application.execute_recognization(cap, process_interval=99, scale_factor=0.33)

    resize_mock.assert_called()
    _, _args, kwargs = resize_mock.mock_calls[0]
    assert kwargs["fx"] == 0.33
    assert kwargs["fy"] == 0.33


def test_execute_recognization_scales_back_coordinates(mocker):
    """
    Agora as coordenadas são reescaladas com (1/scale_factor) e int().
    Ex: top=10 com scale_factor=0.5 => scale_back=2 => top=20.
    """
    mocker.patch(
        "app.application.repository.get_know_face_encodings",
        return_value=([np.zeros(128)], ["andy"])
    )

    cap = mocker.Mock()
    cap.read.side_effect = [
        (True, _fake_frame()),
        (True, _fake_frame()),
        (False, None),
    ]

    mocker.patch("cv2.resize", side_effect=lambda frame, *_args, **_kw: frame)

    mocker.patch("face_recognition.face_locations", return_value=[(10, 20, 30, 40)])
    mocker.patch("face_recognition.face_encodings", return_value=[np.zeros(128)])
    mocker.patch("face_recognition.compare_faces", return_value=[True])
    mocker.patch("face_recognition.face_distance", return_value=[0.0])

    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=0)
    mocker.patch("cv2.destroyAllWindows")

    rect_mock = mocker.patch("cv2.rectangle")

    application.execute_recognization(cap, process_interval=2, scale_factor=0.5)

    # primeira chamada rectangle(frame, (left, top), (right, bottom), color, 2)
    _, args, _ = rect_mock.mock_calls[0]
    pt1 = args[1]  # (left, top)
    pt2 = args[2]  # (right, bottom)

    assert pt1 == (80, 20)   # left=40*2, top=10*2
    assert pt2 == (40, 60) or pt2 == (40, 60)  # right=20*2, bottom=30*2 (ajuste esperado)

def test_get_available_cameras_returns_empty_when_none_available(mocker):
    cap_mock = mocker.Mock()
    cap_mock.isOpened.return_value = False
    mocker.patch("cv2.VideoCapture", return_value=cap_mock)

    result = application.get_available_cameras(max_cameras=5)
    assert result == []

def test_get_available_cameras_returns_indices_and_releases(mocker):
    cap0 = mocker.Mock()
    cap0.isOpened.return_value = True

    cap1 = mocker.Mock()
    cap1.isOpened.return_value = False

    cap2 = mocker.Mock()
    cap2.isOpened.return_value = True

    def side_effect(i):
        return {0: cap0, 1: cap1, 2: cap2}[i]

    mocker.patch("cv2.VideoCapture", side_effect=side_effect)

    result = application.get_available_cameras(max_cameras=3)
    assert result == [0, 2]
    cap0.release.assert_called_once()
    cap2.release.assert_called_once()
    cap1.release.assert_not_called()

def test_get_video_capture_returns_cap_when_camera_opens(mocker):
    cap_mock = mocker.Mock()
    cap_mock.isOpened.return_value = True
    mocker.patch("cv2.VideoCapture", return_value=cap_mock)

    result = application.get_video_capture(0)

    assert result is cap_mock