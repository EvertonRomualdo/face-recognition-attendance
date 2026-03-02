import os
import sys
from colorama import init, Fore, Style
from src.app import application
from src.repository import repository

init(autoreset=True)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    clear_screen()
    print(Fore.CYAN + "=" * 50)
    print(Fore.CYAN + "      SISTEMA DE RECONHECIMENTO FACIAL")
    print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)


def menu_configure_environment(config):
    while True:
        print_header()
        print(Fore.YELLOW + "--- CONFIGURAÇÕES DO AMBIENTE ---" + Style.RESET_ALL)
        print(
            f"1. Forçar Rebuild do Cache : [{Fore.GREEN if config['rebuild_cache'] else Fore.RED}{config['rebuild_cache']}{Style.RESET_ALL}]")
        print(f"2. Intervalo de Processamento: [{Fore.CYAN}{config['process_interval']} frames{Style.RESET_ALL}]")
        print(f"3. Fator de Escala (Resize)  : [{Fore.CYAN}{config['scale_factor']}{Style.RESET_ALL}]")
        print(f"4. Câmera Selecionada        : [{Fore.CYAN}Câmera {config['camera_index']}{Style.RESET_ALL}]")
        print("5. Voltar ao Menu Principal")

        choice = input("\nEscolha uma opção para alterar (1-5): ")

        if choice == '1':
            config['rebuild_cache'] = not config['rebuild_cache']
        elif choice == '2':
            try:
                new_interval = int(input("Novo intervalo de frames (ex: 8): "))
                config['process_interval'] = new_interval
            except ValueError:
                pass
        elif choice == '3':
            try:
                new_scale = float(input("Novo fator de escala (ex: 0.5): "))
                config['scale_factor'] = new_scale
            except ValueError:
                pass
        elif choice == '4':
            print("\nBuscando câmeras...")
            available_cameras = application.get_available_cameras()
            if available_cameras:
                print("Câmeras disponíveis:", available_cameras)
                try:
                    selected_cam = int(input("Digite o índice da câmera: "))
                    if selected_cam in available_cameras:
                        config['camera_index'] = selected_cam
                except ValueError:
                    pass
            else:
                print(Fore.RED + "Nenhuma câmera encontrada." + Style.RESET_ALL)
                input("Pressione Enter...")
        elif choice == '5':
            break

    return config


def menu_add_files(config):
    print_header()
    print(Fore.YELLOW + "--- CADASTRO DE ALUNOS ---" + Style.RESET_ALL)
    print("O explorador de arquivos será aberto para você escolher o vídeo.")

    while True:
        student_name = input("\nDigite o nome do aluno (ou deixe em branco para voltar): ").strip()
        if not student_name:
            break

        print(Fore.CYAN + "Aguardando seleção do arquivo..." + Style.RESET_ALL)
        is_success = repository.import_student_video(student_name)

        if is_success:
            print(Fore.GREEN + f"✅ Sucesso! O vídeo de '{student_name}' foi importado." + Style.RESET_ALL)
            config['rebuild_cache'] = True
            print(Fore.YELLOW + "⚠️ A flag 'Forçar Rebuild do Cache' foi ativada automaticamente." + Style.RESET_ALL)
        else:
            print(Fore.RED + "❌ Importação cancelada ou falhou." + Style.RESET_ALL)

    return config


def menu_execute(config):
    print_header()
    print(Fore.GREEN + "--- INICIANDO SISTEMA ---" + Style.RESET_ALL)

    print(f"Conectando à Câmera {config['camera_index']}...")
    cap = application.get_video_capture(config['camera_index'])

    if cap is None:
        print(Fore.RED + "Erro fatal: Não foi possível acessar a câmera." + Style.RESET_ALL)
        input("\nPressione Enter para voltar...")
        return

    if config['rebuild_cache']:
        print(Fore.YELLOW + "Reconstruindo cache de características (Isso pode demorar)..." + Style.RESET_ALL)
        repository.rebuild_cache()
        config['rebuild_cache'] = False

    print(Fore.CYAN + "Iniciando interface visual... (Pressione 'Q' na janela do vídeo para sair)" + Style.RESET_ALL)

    application.execute_recognization(
        cap,
        process_interval=config['process_interval'],
        scale_factor=config['scale_factor']
    )

    print(Fore.GREEN + "\nSessão encerrada com sucesso." + Style.RESET_ALL)
    input("Pressione Enter para voltar ao menu...")


def main():
    config = {
        "rebuild_cache": False,
        "process_interval": 8,
        "scale_factor": 0.5,
        "camera_index": 0
    }

    while True:
        print_header()
        print("1. Configurar Ambiente")
        print("2. Adicionar Arquivos (Cadastro de Aluno)")
        print("3. Executar Chamada")
        print("4. Sair")

        choice = input("\nSelecione uma opção: ").strip()

        if choice == '1':
            config = menu_configure_environment(config)
        elif choice == '2':
            config = menu_add_files(config)
        elif choice == '3':
            menu_execute(config)
        elif choice == '4':
            clear_screen()
            print(Fore.GREEN + "Encerrando o sistema. Até logo!" + Style.RESET_ALL)
            sys.exit(0)
        else:
            print(Fore.RED + "Opção inválida!" + Style.RESET_ALL)
            input("Pressione Enter para tentar novamente...")


if __name__ == "__main__":
    main()