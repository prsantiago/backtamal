from backtamal.nn import RedNeuronal
from backtamal.nn_cython import RedNeuronal as RedNeuronalCython

if __name__ == "__main__":
    # Test the RedNeuronal class
    red = RedNeuronal(3, [4, 4, 1], fn_act="tanh")
    redC = RedNeuronalCython(3, [4, 4, 1], fn_act="tanh")

    # Test the training method
    X = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    Y = [1.0, -1.0, -1.0, 1.0]
    
    red.entrenamiento(X, Y, epocas=50, lr=0.05)
    print("Python implementation trained")
    redC.entrenamiento(X, Y, epocas=50, lr=0.05)
    print("Cython implementation trained")

    ypred = red.prediccion(X)
    ypredC = redC.prediccion(X)

    print("Prediction from Python implementation:", ypred)
    print("Prediction from Cython implementation:", ypredC)