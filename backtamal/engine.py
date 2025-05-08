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
    __slots__ = ('valor', 'gradiente', '_retroprop', '_previos', '_op', 'etiqueta')

    def __init__(self, valor: float, _hijos=(), _op: str = '', etiqueta: str = ''):
        self.valor: float = float(valor)
        self.gradiente: float = 0.0
        self._retroprop = None
        self._previos = tuple(_hijos)  # tupla, más rápida que set
        self._op: str = _op
        self.etiqueta: str = etiqueta

    def __repr__(self):
        return f"Valor(valor={self.valor})"

    def __add__(self, otro):
        if not isinstance(otro, Valor):
            otro = Valor(otro)
        out = Valor(self.valor + otro.valor, (self, otro), '+')
        def retro():
            self.gradiente += 1.0 * out.gradiente
            otro.gradiente += 1.0 * out.gradiente
        out._retroprop = retro
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, otro):
        if not isinstance(otro, Valor):
            otro = Valor(otro)
        return self + (-otro)

    def __radd__(self, otro):
        return self + otro

    def __mul__(self, otro):
        if not isinstance(otro, Valor):
            otro = Valor(otro)
        out = Valor(self.valor * otro.valor, (self, otro), '*')
        def retro():
            self.gradiente += otro.valor * out.gradiente
            otro.gradiente += self.valor * out.gradiente
        out._retroprop = retro
        return out

    def __rmul__(self, otro):
        return self * otro

    def __pow__(self, exponente):
        assert isinstance(exponente, (int, float)), "El exponente debe ser numérico"
        out = Valor(self.valor ** exponente, (self,), f'**{exponente}')
        def retro():
            self.gradiente += exponente * (self.valor ** (exponente - 1)) * out.gradiente
        out._retroprop = retro
        return out

    def __truediv__(self, otro):
        if not isinstance(otro, Valor):
            otro = Valor(otro)
        return self * (otro ** -1)

    def exp(self):
        x = self.valor
        out = Valor(math.exp(x), (self,), 'exp')
        def retro():
            self.gradiente += out.valor * out.gradiente
        out._retroprop = retro
        return out

    def tanh(self):
        x = self.valor
        t = math.tanh(x)
        out = Valor(t, (self,), 'tanh')
        def retro():
            self.gradiente += (1 - t ** 2) * out.gradiente
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