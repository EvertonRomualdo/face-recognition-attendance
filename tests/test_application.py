from unittest import mock
import numpy as np
import pytest

from app import application


def test_get_video_capture_returns_none_when_camera_fails(mocker):
    fake_cap = mock.Mock()
    fake_cap.isOpened.return_value = False

    mocker.patch("cv2.VideoCapture", return_value=fake_cap)

    cap = application.get_video_capture(0)

    assert cap is None


def test_execute_recognization_registers_presence(mocker):
    fake_cap = mock.Mock()
    fake_cap.read.side_effect = [
        (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        (False, None)
    ]

    mocker.patch(
        "app.application.repository.get_know_face_encodings",
        return_value=([np.zeros(128)], ["Anderson"])
    )

    mocker.patch("face_recognition.face_locations", return_value=[(0, 0, 0, 0)])
    mocker.patch("face_recognition.face_encodings", return_value=[np.zeros(128)])
    mocker.patch("face_recognition.compare_faces", return_value=[True])
    mocker.patch("face_recognition.face_distance", return_value=[0.0])

    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=ord("q"))
    mocker.patch("cv2.destroyAllWindows")
    
    # Mock da função de salvar presença
    save_attendance_mock = mocker.patch("app.application.attendance.save_attendance")

    application.execute_recognization(fake_cap)

    fake_cap.release.assert_called_once()
    # Verifica que a função de salvar presença foi chamada com dados
    assert save_attendance_mock.called
    called_args = save_attendance_mock.call_args[0][0]
    assert "Anderson" in called_args


def test_execute_recognization_processes_every_other_frame(mocker):
    mocker.patch(
        "app.application.repository.get_know_face_encodings",
        return_value=([np.zeros(128)], ["Anderson"])
    )

    cap = mocker.Mock()
    cap.read.side_effect = [
        (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        (False, None),
    ]

    loc_mock = mocker.patch("face_recognition.face_locations", return_value=[(0, 0, 0, 0)])
    enc_mock = mocker.patch("face_recognition.face_encodings", return_value=[np.zeros(128)])
    mocker.patch("face_recognition.compare_faces", return_value=[True])
    mocker.patch("face_recognition.face_distance", return_value=[0.0])

    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=0)
    mocker.patch("cv2.destroyAllWindows")

    application.execute_recognization(cap)

    assert loc_mock.call_count == 1
    assert enc_mock.call_count == 1


@pytest.mark.parametrize(
    "detected_name, expected_color",
    [
        ("Desconhecido", (0, 0, 255)),
        ("Anderson", (0, 255, 0)),
    ],
)
def test_execute_recognization_draws_correct_color(mocker, detected_name, expected_color):
    """
    Testa a regra de cor do retângulo:
    - conhecido: verde
    - desconhecido: vermelho
    """
    mocker.patch(
        "app.application.repository.get_know_face_encodings",
        return_value=([np.zeros(128)], ["Anderson"])
    )

    cap = mocker.Mock()
    cap.read.side_effect = [
        (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        (False, None)
    ]

    mocker.patch("face_recognition.face_locations", return_value=[(1, 2, 3, 4)])
    mocker.patch("face_recognition.face_encodings", return_value=[np.zeros(128)])

    if detected_name == "Anderson":
        mocker.patch("face_recognition.compare_faces", return_value=[True])
        mocker.patch("face_recognition.face_distance", return_value=[0.0])
    else:
        # força não-match
        mocker.patch("face_recognition.compare_faces", return_value=[False])
        mocker.patch("face_recognition.face_distance", return_value=[1.0])

    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=ord("q"))
    mocker.patch("cv2.destroyAllWindows")

    rect_mock = mocker.patch("cv2.rectangle")

    application.execute_recognization(cap)

    assert rect_mock.call_args_list, "cv2.rectangle não f(usamos apenas assert_called pois o loop pode chamar mais de uma vez)oi chamado"
    _, args, _ = rect_mock.mock_calls[0]
    assert args[3] == expected_color


def test_get_available_cameras_returns_empty_when_none_available(mocker):
    cap_mock = mocker.Mock()
    cap_mock.isOpened.return_value = False
    mocker.patch("cv2.VideoCapture", return_value=cap_mock)

    result = application.get_available_cameras(max_cameras=5)
    assert result == []


def test_get_available_cameras_returns_indices_and_releases(mocker):
    """
    Simula câmeras disponíveis nos índices 0 e 2.
    Garante que release() é chamado só nas que abriram.
    """
    cap0 = mocker.Mock()
    cap0.isOpened.return_value = True

    cap1 = mocker.Mock()
    cap1.isOpened.return_value = False

    cap2 = mocker.Mock()
    cap2.isOpened.return_value = True

    def video_capture_side_effect(i):
        return {0: cap0, 1: cap1, 2: cap2}[i]

    mocker.patch("cv2.VideoCapture", side_effect=video_capture_side_effect)

    result = application.get_available_cameras(max_cameras=3)

    assert result == [0, 2]
    cap0.release.assert_called_once()
    cap2.release.assert_called_once()
    cap1.release.assert_not_called()
