import torchaudio.functional as F
import torchaudio
import torch
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import hamming
import soundfile as sf

import sys
print("Current env " + sys.prefix)

# Load a WAV file
signal, signal_rate = sf.read(
    "C:\\Users\\temoB\\OneDrive\\Documents\\0 2024\\MX\\UNAM\\Reconocimiento de patrones\\practica2\\a.wav")
# rate, waveform = wavfile.read('./a.wav')

plt.show


def beautyPlot(signal, title, xlabel='Time', ylabel='Amplitude'):
    # Plot the original and resampled signals
    plt.figure(figsize=(10, 6))
    plt.plot(signal, 'b', label='Original signal')
    # plt.plot(x_resampled, y_resampled, 'or-', label='Resampled signal')
    plt.legend(loc='lower left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# Ejercicio 1. Gráfica de la señal
# beautyPlot(signal, title='Voice for vocal A')


# Ejercicio 2. Aplicado de preénfasis
def preenfasis(signal, a=0.9):
    sn = np.zeros(len(signal))
    for n in range(1, len(signal)):
        sn[n] = signal[n] - a*signal[n - 1]
    return sn


psn = preenfasis(signal)
# beautyPlot(psn, title='Preenphasis - Voice for vocal A')
# Puede notarse cómo el filtro de preenfasis elimino ciertas frecuencias, en el dominio del tiempo puede verse que las amplitudes han sido redusidas o suavizadas.

# Ejercicio 3. Ventana de Hamming
hpsn = hamming.hamming_window(psn)
