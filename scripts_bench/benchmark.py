import time
import math
import numpy as np
import face_recognition
import matplotlib.pyplot as plt

def benchmark(n_known_faces, n_runs=100):
    known_encodings = [np.random.rand(128) for _ in range(n_known_faces)]
    test_encoding = np.random.rand(128)

    times = []

    for _ in range(n_runs):
        start = time.perf_counter()

        face_recognition.compare_faces(known_encodings, test_encoding)
        face_recognition.face_distance(known_encodings, test_encoding)

        end = time.perf_counter()
        times.append(end - start)

    mean = np.mean(times)
    std = np.std(times)
    ic95 = 1.96 * (std / math.sqrt(n_runs))

    return mean, std, ic95

if __name__ == "__main__":
    n_values = [5, 20, 50, 100, 200]
    mean_times = []

    print("===== RESULTADOS DO BENCHMARK =====")
    for n in n_values:
        mean, std, ic = benchmark(n)
        mean_times.append(mean)
        print(
            f"N = {n} → média = {mean:.6f}s | desvio = {std:.6f}s | IC95% = ±{ic:.6f}s"
        )

    plt.figure()
    plt.plot(n_values, mean_times, marker='o')
    plt.xlabel("Número de rostos cadastrados (N)")
    plt.ylabel("Tempo médio de comparação (s)")
    plt.title("Crescimento do Tempo de Comparação Facial")
    plt.grid(True)
    plt.show()