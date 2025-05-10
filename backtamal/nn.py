import random

from backtamal.engine import Valor


class Neurona:

    def __init__(self, nin):
        self.w = [Valor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Valor(random.uniform(-1, 1))

    def __call__(self, x):
        assert len(x) == len(self.w), "El tamaño de x no coincide con el número de pesos"
        
        out = self.b
        out += sum(w * x_i for w, x_i in zip(self.w, x))

        return out.tanh()
    
class Capa:
    def __init__(self, nin, nout):
        self.neuronas = [Neurona(nin) for _ in range(nout)]
    
    def __call__(self, x):
        o = [neurona(x) for neurona in self.neuronas]

        return o[0] if len(o) == 1 else o
    

class RedNeuronal:
    def __init__(self, nin, ncapas):
        sz = [nin] + ncapas
        self.capas = [Capa(sz[i], sz[i+1]) for i in range(len(ncapas))]
    
    def __call__(self, x):
        assert isinstance(x, list), "x debe ser una lista"
        assert len(x) == len(self.capas[0].neuronas[0].w), "El tamaño de x no coincide con el número de pesos"

        for capa in self.capas:
            x = capa(x)
        
        return x