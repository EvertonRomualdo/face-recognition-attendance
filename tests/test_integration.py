import numpy as np
from app import application

def _fake_frame():
    return np.zeros((480, 640, 3), dtype = np.uint8)

def test_integration_full_flow_two_detections_registers(mocker):
    mock_repo = mocker.patch(
        "app.application.repository.get_know_face_encodings",
        return_value=([np.zeros(128)], ["Anderson"])
    )

    cap = mocker.Mock()
    cap.read.side_effect = [
        (True, _fake_frame()),
        (True, _fake_frame()),
        (False, None)
    ]

    mocker.patch("cv2.resize", side_effect=lambda frame, *_a, **_k: frame)
    mocker.patch("face_recognition.face_locations", return_value=[(10, 40, 40, 10)])
    mocker.patch("face_recognition.face_encodings", return_value=[np.zeros(128)])

    dist_mock = mocker.patch("face_recognition.face_distance", return_value=np.array([0.0]))

    mocker.patch("app.application.cv2.imshow")
    mocker.patch("app.application.cv2.waitKey", return_value=0)
    mocker.patch("app.application.cv2.destroyAllWindows")
    mocker.patch("app.application.cv2.namedWindow")
    mocker.patch("app.application.cv2.setWindowProperty")

    save_mock = mocker.patch("app.application.attendance.save_attendance")

    # Execução
    application.execute_recognization(cap, process_interval=1, scale_factor=0.5)
    cap.release.assert_called_once()
    
    cap.release.assert_called_once()
    assert dist_mock.call_count >= 1
    save_mock.assert_called_once()
    mock_repo.assert_called_once()

def test_integration_unknown_face_not_save(mocker):
    mock_repo = mocker.patch(
        "app.application.repository.get_know_face_encodings",
        return_value=([np.zeros(128)], ["Anderson"])
    )

    cap = mocker.Mock()
    cap.read.side_effect = [
        (True, _fake_frame()),
        (False, None)
    ]
    mocker.patch("cv2.resize", side_effect=lambda frame, *_a, **_k: frame)
    mocker.patch("face_recognition.face_locations", return_value=[(10, 40, 40, 10)])
    mocker.patch("face_recognition.face_encodings", return_value=[np.ones(128)])

    dist_mock = mocker.patch("face_recognition.face_distance", return_value=np.array([0.9]))

    mocker.patch("app.application.cv2.imshow")
    mocker.patch("app.application.cv2.waitKey", return_value=ord("q"))
    mocker.patch("app.application.cv2.destroyAllWindows")
    mocker.patch("app.application.cv2.namedWindow")
    mocker.patch("app.application.cv2.setWindowProperty")

    save_mock = mocker.patch("app.application.attendance.save_attendance")

    # Execução
    application.execute_recognization(cap, process_interval=1, scale_factor=0.5)

    cap.release.assert_called_once()
    mock_repo.assert_called_once()
    assert dist_mock.call_count >= 1
    save_mock.assert_not_called()