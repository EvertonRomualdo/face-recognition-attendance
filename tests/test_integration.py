import numpy as np
from app import application

def test_integration_full_flow_one_frame(mocker):
    mock_repo = mocker.patch(
        "app.application.repository.get_know_face_encodings",
        return_value=([np.zeros(128)], ["Anderson"])
    )

    cap = mocker.Mock()
    cap.read.side_effect = [
        (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        (False, None)
    ]

    mock_locations = mocker.patch("face_recognition.face_locations", return_value=[(10, 40, 40, 10)])
    mock_encodings = mocker.patch("face_recognition.face_encodings", return_value=[np.zeros(128)])
    mock_compare = mocker.patch("face_recognition.compare_faces", return_value=[True])
    mocker.patch("face_recognition.face_distance", return_value=[0.0])
    mock_imshow = mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=ord('q'))
    mocker.patch("cv2.destroyAllWindows")

    # Execução
    application.execute_recognization(cap)
    cap.release.assert_called_once()
    
    mock_repo.assert_called_once()        # Garante que buscou os alunos no banco
    mock_locations.assert_called_once()   # Garante que tentou encontrar rostos no frame
    mock_encodings.assert_called_once()   # Garante que extraiu as características do rosto
    mock_compare.assert_called_once()     # Garante que comparou o rosto encontrado com o banco
    mock_imshow.assert_called()           # Garante que tentou exibir a imagem