import random

from backtamal.engine import Valor

class Modulo:

    def grad_cero(self):
        for p in self.parametros():
            p.gradiente = 0

    def parametros(self):
        return []

class Neurona(Modulo):

    def __init__(self, nin, nolineal=True, fn_act="relu"):
        self.w = [Valor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Valor(random.uniform(-1, 1))
        self.fn_act = fn_act
        self.nolineal = nolineal

    def __call__(self, x):
        assert len(x) == len(self.w), "El tamaño de x no coincide con el número de pesos"
        
        out = self.b
        out += sum(w_i * x_i for w_i, x_i in zip(self.w, x))

        # FIXME: Not the best way to do this
        if self.fn_act == "relu":
            out = out.relu()
        elif self.fn_act == "tanh":
            out = out.tanh()

        return out
    
    def parametros(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"Neurona(w={self.w}, b={self.b}, nolineal={self.nolineal})"
    

class Capa(Modulo):

    def __init__(self, nin, nout, **kwargs):
        self.neuronas = [Neurona(nin, **kwargs) for _ in range(nout)]
    
    def __call__(self, x):
        o = [neurona(x) for neurona in self.neuronas]

        return o[0] if len(o) == 1 else o
    
    def parametros(self):
        params = []
        for neurona in self.neuronas:
            params.extend(neurona.parametros())
        return params
    
    def __repr__(self):
        return f"Capa(neuronas=[{', '.join(str(n) for n in self.neurons)}]"

 
class RedNeuronal(Modulo):

    def __init__(self, nin, ncapas):
        sz = [nin] + ncapas
        self.capas = [Capa(sz[i], sz[i+1], nolineal=i!=len(ncapas)-1) for i in range(len(ncapas))]
    
    def __call__(self, x):
        assert isinstance(x, list), "x debe ser una lista"
        assert len(x) == len(self.capas[0].neuronas[0].w), "El tamaño de x no coincide con el número de pesos"

        for capa in self.capas:
            x = capa(x)
        
        return x
    
    def parametros(self):
        params = []
        for capa in self.capas:
            params.extend(capa.parametros())
        return params
    
    def entrenamiento(self, x, y, epocas=10, lr=0.01):
        for epoca in range(epocas):
            
            # Paso hacia adelante
            ypred = [self(xi) for xi in x]
            perd = sum((yout - ygt) ** 2 for ygt, yout in zip(y, ypred))
            
            # Paso hacia atras (descenso del gradiente)
            perd.backward()

            # Actualizar pesos y reiniciar gradientes
            for p in self.parametros():
                p.dato -= lr * p.gradiente
                p.gradiente = 0

            # Imprimir resultados cada 5 epocas
            if epoca % 5 == 0:
                print(f"Epoch {epoca}/{epocas}, Loss: {perd.dato:.4f}")

    def prediccion(self, x):
        return [self(xi) for xi in x]
    
    def __repr__(self):
        return f"RedNeuronal(capacidades=[{', '.join(str(layer) for layer in self.layers)}])"
