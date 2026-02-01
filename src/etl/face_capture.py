import cv2
import numpy as np
import os
import face_recognition
from pathlib import Path


def get_image():
    source = Path(__file__).resolve().parent.parent.parent
    imagedir = source / "data" / "raw_images"
    images = {}
    for file in imagedir.iterdir():
        name = os.path.splitext(file.name)[0]

        images[name] = str(file)
    print(images)
    return images

def capture_video():
    cap = cv2.VideoCapture(0)
    images_dict = get_image()
    know_face_encoding = []
    know_face_names = []

    #pegos as imagens em numpy array
    #imagen não interresa, so enconding e nome
    for img in images_dict:
        print(img)
        image_array = face_recognition.load_image_file(images_dict.get(img))
        know_face_encoding.append(face_recognition.face_encodings(image_array)[0])
        know_face_names.append(img)


    students = know_face_names.copy()

    #listas das capturas de camera
    face_location = []
    face_encoding = []
    face_names = []

    '''
        Lista de problemas/otimizações:
            * É necessario um sistema de "cache" com encodings ja processados para ganhar em tempo
            * A atual implementação esta bagunçada e não muito "arquitetada"
            * Atualmente o codigo limita a deteção de uma face por vez. O np.argmin é o responsavel
            * O codigo pode ficar mais limpo e distribuido
            * TENHO QUE OTIMIZAR! o codigo simplesmente não roda no meu PC :(
    '''
    while True:
        _, frame = cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = np.ascontiguousarray(rgb_small_frame)
        if True:
            #todas as faces que estão na camera
            face_location = face_recognition.face_locations(rgb_small_frame)
            #faço o encoding do frame e passo a posição das faces
            face_encoding = face_recognition.face_encodings(rgb_small_frame, face_location)
            face_names = []
            for face_encoding in face_encoding:
                #retorna um array booleano se da match no rosto ou não
                matches = face_recognition.compare_faces(know_face_encoding, face_encoding)
                name = ""
                #distancia euclidiana que diz o quão similares a faces são na mesma ordem de know_faces
                #quanto menor mais parecida
                face_distances = face_recognition.face_distance(know_face_encoding, face_encoding)
                #pega a que mais se parece


                #Aqui eu limito a deteção a apeas uma face!!
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = know_face_names[best_match_index]
                print(f"Rosto detectado: {name}")

                face_names.append(name)
                if name in know_face_names:
                    if name in students:
                        students.remove(name)
                        print(name)
                        print(students)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()









