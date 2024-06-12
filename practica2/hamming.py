import math
import numpy as np


def hamming_window(signal, window_size=512, window_phase=170):
    seq = [*range(0, window_size)]
    print(seq)
    # seq2 = np.arange(window_size) # type array([])

    # w = [0]*window_size
    # for i in range(0, window_size):
    #     w[i] = 0.54 - 0.4 * \
    #         math.cos(2*math.pi*seq[i]/(window_size - 1))

    w = [0.54 - 0.46 *
         math.cos(2*math.pi*seq[i]/(window_size - 1)) for i in range(0, window_size)]

    sh = []
    for j in range(0, len(signal) - window_size, window_phase):
        part = np.array(signal[j: j + window_size])
        result = part * w
        sh.extend(result)

    h = [np.array(signal[j: j + window_size]) *
         w for j in range(0, len(signal) - window_size, window_phase)]

    return h, sh
