import time
from backtamal.nn import RedNeuronal as RedNeuronalPy
from backtamal.nn_cython import RedNeuronal as RedNeuronalCython

def test_red_neuronal():
    x = [5.0, 1.0, -3.0, 0.5, 10.0]
    n = RedNeuronalPy(5, [4, 4, 1])
    out = n(x)
    print("Salida RedNeuronal (Python):", out)

def test_red_neuronal_cython():
    x = [5.0, 1.0, -3.0, 0.5, 10.0]
    n = RedNeuronalCython(5, [4, 4, 1])
    out = n(x)
    print("Salida RedNeuronal (Cython):", out)

def benchmark_red(N=1000):
    x = [5.0, 1.0, -3.0, 0.5, 10.0]
    n_py = RedNeuronalPy(5, [4, 4, 1])
    n_cy = RedNeuronalCython(5, [4, 4, 1])

    # Python
    t0 = time.time()
    for _ in range(N):
        n_py(x)
    t1 = time.time()
    print(f"RedNeuronal Python: {t1-t0:.4f} segundos")

    # Cython
    t2 = time.time()
    for _ in range(N):
        n_cy(x)
    t3 = time.time()
    print(f"RedNeuronal Cython: {t3-t2:.4f} segundos")

if __name__ == "__main__":
    print("== Test Python ==")
    test_red_neuronal()
    print("\n== Test Cython ==")
    test_red_neuronal_cython()
    print("\n== Benchmark ==")
    benchmark_red()
