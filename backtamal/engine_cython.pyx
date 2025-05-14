import math

class Valor:
    """
    Representa un valor escalar y su gradiente para cálculo automático.
    Permite construir grafos computacionales y realizar backpropagation.

    - Usa __slots__ para reducir memoria y acelerar acceso.
    - Evita lambdas y funciones anidadas, usando funciones globales.
    - Usa float puro para atributos.
    - Usa tuplas para _previos (más rápido que set).
    - Prepara el código para fácil compilación con Cython.
    """
    __slots__ = ('dato', 'gradiente', '_retroprop', '_previos', '_op', 'etiqueta')

    def __init__(self, dato: float, _hijos=(), _op: str = '', etiqueta: str = ''):
        self.dato: float = dato
        self.gradiente: float = 0.0
        self._retroprop = None
        self._previos = tuple(_hijos)  # tupla, más rápida que set
        self._op: str = _op
        self.etiqueta: str = etiqueta

    def __repr__(self):
        return f"Valor(dato={self.dato}) gradiente={self.gradiente}"

    def __add__(self, otro):
        otro = otro if isinstance(otro, Valor) else Valor(otro)
        out = Valor(self.dato + otro.dato, (self, otro), '+')

        def retro():
            self.gradiente += 1.0 * out.gradiente
            otro.gradiente += 1.0 * out.gradiente
        out._retroprop = retro

        return out
    
    def __mul__(self, otro):
        otro = otro if isinstance(otro, Valor) else Valor(otro)
        out = Valor(self.dato * otro.dato, (self, otro), '*')

        def retro():
            self.gradiente += otro.dato * out.gradiente
            otro.gradiente += self.dato * out.gradiente
        out._retroprop = retro

        return out

    def __pow__(self, exponente):
        assert isinstance(exponente, (int, float)), "El exponente debe ser numérico"
        out = Valor(self.dato ** exponente, (self,), f'**{exponente}')

        def retro():
            self.gradiente += (exponente * (self.dato ** (exponente - 1))) * out.gradiente
        out._retroprop = retro

        return out

    def exp(self):
        x = self.dato
        out = Valor(math.exp(x), (self,), 'exp')
        def retro():
            self.gradiente += out.dato * out.gradiente
        out._retroprop = retro
        return out

    def tanh(self):
        x = self.dato
        t = math.tanh(x)
        out = Valor(t, (self,), 'tanh')
        def retro():
            self.gradiente += (1 - t ** 2) * out.gradiente
        out._retroprop = retro
        return out

    def relu(self):
        out = Valor(max(0, self.dato), (self,), 'relu')

        def retro():
            if self.dato > 0:
                self.gradiente += out.gradiente
        out._retroprop = retro

        return out

    def backward(self):
        topo = []
        visitados = set()
        def construir_topo(valor):
            if valor not in visitados:
                visitados.add(valor)
                for hijo in valor._previos:
                    construir_topo(hijo)
                topo.append(valor)
        construir_topo(self)
        self.gradiente = 1.0
        for nodo in reversed(topo):
            if nodo._retroprop:
                nodo._retroprop()
        
    def __neg__(self):
        return self * -1

    def __radd__(self, otro):
        return self + otro

    def __sub__(self, otro):
        return self + (-otro)
    
    def __rsub__(self, otro):
        return otro + (-self)

    def __rmul__(self, otro):
        return self * otro

    def __truediv__(self, otro):
        return self * (otro ** -1)

    def __rtruediv__(self, otro):
        return otro * (self ** -1)