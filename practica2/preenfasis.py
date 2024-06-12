import numpy as np

# Ejercicio 2. Aplicado de preénfasis


def preenfasis(signal, a=0.9):
    # sn = np.zeros(len(signal))
    # for n in range(1, len(signal)):
    #     sn[n] = signal[n] - a*signal[n - 1]
    return [signal[n] - a*signal[n - 1] for n in range(1, len(signal))]
