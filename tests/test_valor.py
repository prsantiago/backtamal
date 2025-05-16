import time
from backtamal.engine import Valor
from backtamal.engine_cython import Valor as ValorCython

def visualizar_grafo(nodo, visitados=None, nivel=0):
    """
    Imprime el grafo de operaciones a partir de un nodo.
    """
    if visitados is None:
        visitados = set()
    if nodo in visitados:
        return
    visitados.add(nodo)
    print('  ' * nivel + f"{nodo.etiqueta or nodo._op or 'dato'}: dato={nodo.dato}, gradiente={nodo.gradiente}")
    for hijo in getattr(nodo, '_previos', []):
        visualizar_grafo(hijo, visitados, nivel + 1)

def forward_backward_neurona(x1, x2, w1, w2, b):
    """
    Simula el forward y backward de una neurona artificial:
    n = x1*w1 + x2*w2 + b
    o = tanh(n)
    Realiza retropropagación desde o.
    Retorna la salida o.
    """
    n = x1 * w1 + x2 * w2 + b
    o = n.tanh()
    o.retropropagacion()
    return o

def benchmark(x1, x2, w1, w2, b, x1c, x2c, w1c, w2c, bc, N=10000):
    """
    Compara el tiempo de ejecución entre Valor, ValorCython usando la operación de neurona artificial.
    """
    # Valor
    t0 = time.time()
    for _ in range(N):
        out = forward_backward_neurona(x1, x2, w1, w2, b)
        x1.gradiente = x2.gradiente = w1.gradiente = w2.gradiente = b.gradiente = 0.0
    t1 = time.time()
    print(f"Valor: {t1-t0:.4f} segundos")

    # ValorCython
    t2 = time.time()
    for _ in range(N):
        out_super_cython = forward_backward_neurona(x1c, x2c, w1c, w2c, bc)
        x1c.gradiente = x2c.gradiente = w1c.gradiente = w2c.gradiente = bc.gradiente = 0.0
    t3 = time.time()
    print(f"ValorCython: {t3-t2:.4f} segundos")

if __name__ == "__main__":
    # Declaración de variables para Valor
    x1 = Valor(3.0, etiqueta='x1')
    x2 = Valor(1.0, etiqueta='x2')
    w1 = Valor(-1.0, etiqueta='w1')
    w2 = Valor(0.5, etiqueta='w2')
    b = Valor(3.0, etiqueta='b')

    x1c = ValorCython(3.0, etiqueta='x1')
    x2c = ValorCython(1.0, etiqueta='x2')
    w1c = ValorCython(-1.0, etiqueta='w1')
    w2c = ValorCython(0.5, etiqueta='w2')
    bc = ValorCython(3.0, etiqueta='b')

    # Visualización de grafo para Valor
    salida = forward_backward_neurona(x1, x2, w1, w2, b)
    print("Grafo para Valor:")
    visualizar_grafo(salida)

    print("\nBenchmark de velocidad:")
    benchmark(x1, x2, w1, w2, b, x1c, x2c, w1c, w2c, bc)
