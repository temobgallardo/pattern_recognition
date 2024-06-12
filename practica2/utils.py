import numpy as np
import matplotlib.pyplot as plt


def pprintm(m, sep="\t"):
    for r in m:
        print(*r, sep)


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


def beautyPlot2(x, y, title, xlabel='Time', ylabel='Amplitude'):
    # Plot the original and resampled signals
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b', label='Original signal')
    # plt.plot(x_resampled, y_resampled, 'or-', label='Resampled signal')
    plt.legend(loc='lower left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def power(x):
    p = 0
    for i in range(len(x)):
        p += np.abs(x[i]**2)
    return p/len(x)
