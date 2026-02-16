from app import application

available_cameras = application.get_available_cameras()

if not available_cameras:
    print("Erro: Nenhuma câmera detectada no sistema.")
    exit(1)

# Lista as câmeras disponíveis
print("Câmeras disponíveis:")
for idx, cam_id in enumerate(available_cameras):
    print(f"{idx + 1}: Câmera {cam_id}")

'''
por enquanto esta usando o terminal para escolher a câmera,
mas futuramente pode ser implementado uma interface gráfica para isso (lista dropdown ou oq for melhor).
'''

# Solicita escolha do usuário
while True:
    try:
        choice = int(input("Escolha o número da câmera (1 a {}): ".format(len(available_cameras))))
        if 1 <= choice <= len(available_cameras):
            selected_camera = available_cameras[choice - 1]
            break
        else:
            print("Escolha inválida. Tente novamente.")
    except ValueError:
        print("Entrada inválida. Digite um número.")

# Usa a câmera selecionada
cap = application.get_video_capture(selected_camera)
if cap is None:
    print("Erro: Não foi possível abrir a câmera selecionada.")
    exit(1)

application.execute_recognization(cap)