from backtamal.nn import RedNeuronal
from backtamal.nn_cython import RedNeuronal as RedNeuronalCython

if __name__ == "__main__":
    # Test the RedNeuronal class
    red = RedNeuronal(5, [4, 4, 1], fn_act="tanh")
    redC = RedNeuronalCython(5, [4, 4, 1], fn_act="tanh")

    # Test the training method
    X = [[2.5, 1.0, -1.1, 9.8, -1.2], [-3.2, 1.1, 2.0, -1.6, 7.5], [1.5, 5.5, 1.1, 9.0, 1.3], [8.8, 7.7, 1.2, 4.0, -5.0]]
    Y = [1.0, -1.0, -1.0, 1.0]
    
    red.entrenamiento(X, Y, epocas=50, lr=0.05)
    print("Python implementation trained")
    redC.entrenamiento(X, Y, epocas=50, lr=0.05)
    print("Cython implementation trained")

    ypred = red.prediccion(X)
    ypredC = redC.prediccion(X)

    print("Prediction from Python implementation:", ypred)
    print("Prediction from Cython implementation:", ypredC)