import cv2
import numpy as np
import face_recognition
import repository

'''
Lista de problemas/otimizações (ORIGINAIS):
    * É necessario um sistema de "cache" com encodings ja processados para ganhar em tempo
    * A atual implementação esta bagunçada e não muito "arquitetada"
    * O codigo pode ficar mais limpo e distribuido
    * TENHO QUE OTIMIZAR! o codigo simplesmente não roda no meu PC :(
'''


def execute_recognization(cap: cv2.VideoCapture):

    # carrega os dados
    print("Carregando faces conhecidas...")
    known_face_encodings, known_face_names = repository.get_know_face_encodings()

    students_missing = known_face_names.copy()

    print(f"{len(known_face_names)} rostos carregados.")

    # Variáveis de estado
    process_this_frame = True
    face_locations = []
    face_encodings = []
    face_names = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Otimizacao: o tamanho do frame foi reduzido para acelerar processamento
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Correcao de cores BGR para RGB e memória continua
        rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = np.ascontiguousarray(rgb_small_frame)

        # Otmização: processa um quadro sim outro nao
        if process_this_frame:
            # Detecta posições
            face_locations = face_recognition.face_locations(rgb_small_frame)
            # Cria encodings
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Desconhecido"

                # Calcula a distância euclidiana. Quanto menor mais parecido
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                face_names.append(name)

                # Lógica de Registro de Presença
                if name in students_missing:
                    students_missing.remove(name)
                    print(f"✅ PRESENÇA REGISTRADA: {name}")
                    print(f"   Faltam: {students_missing}")

        # inverse a flag
        process_this_frame = not process_this_frame

        # desenha o retangulo no rosto
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Cor: Verde se conhecido, Vermelho se desconhecido
            color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow('Reconhecimento Facial', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_video_capture(cam_ip=0):
    cap = cv2.VideoCapture(cam_ip)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera")
        return None

    return cap
